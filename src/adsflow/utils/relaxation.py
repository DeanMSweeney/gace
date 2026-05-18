import contextlib
import logging
import os
import shutil
import subprocess
import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE
from fairchem.core import OCPCalculator

from huggingface_hub import hf_hub_download

from adsflow.utils.atoms import data_to_atoms
from fairchem.core.common.relaxation.optimizable import OptimizableBatch


class _PatchedOptimizableBatch(OptimizableBatch):
    """Adds the ASE >= 3.23 Optimizable abstract methods missing from fairchem's OptimizableBatch."""

    def get_x(self):
        return self.get_positions().reshape(-1)

    def set_x(self, x):
        self.set_positions(x.reshape(-1, 3))

    def get_gradient(self):
        return -self.get_forces().reshape(-1)

    def get_value(self):
        return self.get_potential_energy()

    def ndofs(self):
        return len(self)

# OC20 gas-phase reference energies (eV) from H2O/N2/CO/H2 linear combination
ATOMIC_REFERENCE_ENERGIES = {
    "H": -3.477,
    "N": -8.083,
    "O": -7.204,
    "C": -7.282,
}

class Silent:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Save original file descriptors
        self._original_stdout_fd = os.dup(1)
        self._original_stderr_fd = os.dup(2)

        # Redirect low-level file descriptors
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore low-level file descriptors
        os.dup2(self._original_stdout_fd, 1)
        os.dup2(self._original_stderr_fd, 2)
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)

        # Restore Python streams
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def config_DFT(
    data: Data,
    gen_POS: str,
    in_POS: Optional[str]=None, 
    rlxd_POS: Optional[str]=None,
):
    print(">>> Running VASP DFT calculations w/ user provided input files  🖲️")
    # Get the current working directory to return to later
    original_dir = os.getcwd()

    try:
        # Get the sid from the data object
        sid = data.sid.item()

        # Create the strucs directory if it doesn't exist
        if not os.path.exists("./strucs"):
            os.makedirs("./strucs")

        # Create the sid directory within strucs
        sid_dir = os.path.join("./strucs", str(sid))
        os.makedirs(sid_dir, exist_ok=True)
        
        # Function to build simulation directory
        def build_vasp_dir(sim_dir: Optional[bool]=None, struc_file: Optional[bool]=None):
            sim_path = os.path.join(sid_dir, sim_dir)
            os.makedirs(sim_path, exist_ok=True)
            shutil.move(struc_file, os.path.join(sim_path, "POSCAR"))
            
            for file in ["INCAR", "KPOINTS", "VASP.sh"]:
                if os.path.exists(file):
                    shutil.copy(file, sim_path)
                else: 
                    raise AttributeError(f"Missing {file} file for VASP run.")

            # Submit the .sh file in each directory
            os.chdir(sim_path)
            subprocess.run(["potcar.sh"])
            os.chdir(original_dir)
            
        # Move and rename files to their respective directories
        if os.path.exists(gen_POS):
            build_vasp_dir(sim_dir="generated", struc_file=gen_POS)
        else: 
            raise AttributeError("Generated strucutre file does not exist")
        
        if os.path.exists(in_POS):
            build_vasp_dir(sim_dir="inputted", struc_file=in_POS)
            
        if os.path.exists(rlxd_POS) and hasattr(data, "pos_relaxed"):
            build_vasp_dir(sim_dir="relaxed", struc_file=rlxd_POS)
            
    finally:
        # Always return to the original directory
        os.chdir(original_dir)
   
def relax_w_mlff(
    atoms: Atoms,
    calculator: OCPCalculator,
    fmax: float = 0.05,
    steps: int = 200,
    optimizer: str = "BFGS",
) -> tuple[Atoms, bool, int]:
    """
    Relax an atomic structure using the MLFF calculator.

    Args:
        atoms: ASE Atoms object to relax
        calculator: OCP calculator with MLFF model
        fmax: Force convergence criterion (eV/A)
        steps: Maximum optimization steps
        optimizer: Optimizer type ("BFGS", "LBFGS", or "FIRE")

    Returns:
        Tuple of (relaxed_atoms, converged, num_steps)
    """
    atoms = atoms.copy()
    atoms.calc = calculator

    # Select optimizer
    optimizers = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE,
    }
    opt_class = optimizers.get(optimizer, BFGS)

    # Run optimization
    opt = opt_class(atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)

    if bool(converged) == False:
        atoms = None

    return atoms


def batch_relax_w_mlff(
    atoms_list: list[Atoms],
    calculator: OCPCalculator,
    fmax: float = 0.05,
    steps: int = 200,
) -> list[Optional[Atoms]]:
    """Relax a list of structures in parallel using fairchem's batched LBFGS optimizer.

    Args:
        atoms_list: ASE Atoms objects to relax.
        calculator: OCPCalculator wrapping the MLFF trainer.
        fmax: Force convergence criterion (eV/Å).
        steps: Max optimisation steps.

    Returns:
        List of relaxed Atoms (SinglePointCalculator attached) or None per entry on failure.
    """
    import fairchem.core.common.relaxation.ml_relaxation as _ml_relax_mod
    from fairchem.core.common.relaxation.ml_relaxation import ml_relax
    from fairchem.core.common.relaxation.ase_utils import batch_to_atoms
    from fairchem.core.preprocessing import AtomsToGraphs
    from torch_geometric.data import Batch as PygBatch

    _ml_relax_mod.OptimizableBatch = _PatchedOptimizableBatch

    # r_edges=False → OptimizableBatch sets otf_graph=True and rebuilds edges each step
    a2g = AtomsToGraphs(
        r_energy=False, r_forces=False, r_distances=False, r_edges=False, r_fixed=True
    )
    data_list = [a2g.convert(atoms, sid=i) for i, atoms in enumerate(atoms_list)]
    batch = PygBatch.from_data_list(data_list)

    try:
        with Silent():
            relaxed_batch = ml_relax(
                batch=batch,
                model=calculator.trainer,
                steps=steps,
                fmax=fmax,
            )
        results = {
            "energy": relaxed_batch.y,
            "forces": relaxed_batch.force,
        }
        relaxed_atoms = batch_to_atoms(relaxed_batch, results=results)

        natoms = relaxed_batch.natoms.tolist()
        forces_per_structure = torch.split(relaxed_batch.force, natoms)
        for i, f in enumerate(forces_per_structure):
            max_force = f.norm(dim=-1).max().item()
            if max_force > fmax:
                logging.warning(f"structure {i} did not converge: max_force={max_force:.4f} > fmax={fmax}")
                relaxed_atoms[i] = None

        return relaxed_atoms
    except Exception as e:
        logging.warning(f"batch_relax_w_mlff failed: {e}")
        return [None] * len(atoms_list)


_CSV_DECIMALS = 6
_FMT = f".{_CSV_DECIMALS}f"


def _fmt(v: Optional[float]) -> str:
    return format(v, _FMT) if v is not None else ""


def relax_struc(
    atoms: Atoms,
    calculator: OCPCalculator,
    config_vasp: bool = False,
) -> tuple:
    """
    Post-generation relaxation step.

    Applies MLFF relaxation.

    Args:
        atoms: ASE Atoms (slab + adsorbate; tags 1=slab, 2=adsorbate)
        calculator: OCP calculator with loaded MLFF model
        run_mlff: Whether to run MLFF relaxation
        config_vasp: Whether to configure VASP DFT input files

    Returns:
        (atoms_relaxed, energy_dict)
    """

    atoms_relaxed = relax_w_mlff(
        atoms=atoms,
        calculator=calculator,
        fmax=0.05,
        steps=200,
        optimizer="BFGS",
    )

    try:
        E_slab_ads = atoms_relaxed.get_potential_energy() 

        return atoms_relaxed, {
            "computed_binding_E": E_slab_ads,
        }
    except Exception as e:
        logging.warning(f">>> Failed to calculate binding energy due to failed relaxation 😢 {e}")
        return atoms_relaxed, None
