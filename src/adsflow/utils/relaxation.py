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

    return atoms, converged, opt.nsteps


_CSV_DECIMALS = 6
_FMT = f".{_CSV_DECIMALS}f"


def _fmt(v: Optional[float]) -> str:
    return format(v, _FMT) if v is not None else ""


def relax_struc(
    atoms: Atoms,
    calculator: OCPCalculator,
    run_mlff: bool = False,
    config_vasp: bool = False,
    calc_binding_energy: bool = False,
) -> tuple:
    """
    Post-generation relaxation step.

    Applies MLFF relaxation. When calc_binding_energy=True, also relaxes the
    isolated slab and computes the OC20 adsorption energy.

    Adsorption energy: E_ads = E(slab+ads) - E(bare_slab) - E(ads_ref)
    where E(ads_ref) = sum of ATOMIC_REFERENCE_ENERGIES for each adsorbate atom.

    Args:
        atoms: ASE Atoms (slab + adsorbate; tags 1=slab, 2=adsorbate)
        calculator: OCP calculator with loaded MLFF model
        run_mlff: Whether to run MLFF relaxation
        config_vasp: Whether to configure VASP DFT input files
        calc_binding_energy: If True, compute adsorption energy via two relaxations

    Returns:
        (atoms_relaxed, energy_dict) where energy_dict is None when
        calc_binding_energy=False or the calculation fails. energy_dict keys:
        slab_ads_energy, slab_energy, ads_ref_energy, ads_energy.
    """
    if not run_mlff:
        return None, None

    atoms_relaxed, _, _ = relax_w_mlff(
        atoms=atoms,
        calculator=calculator,
        fmax=0.05,
        steps=200,
        optimizer="BFGS",
    )

    if not calc_binding_energy:
        return atoms_relaxed, None

    try:
        E_slab_ads = atoms_relaxed.get_potential_energy()

        slab_atoms = atoms[atoms.get_tags() != 2]
        slab_relaxed, _, _ = relax_w_mlff(
            atoms=slab_atoms,
            calculator=calculator,
            fmax=0.05,
            steps=200,
            optimizer="BFGS",
        )
        E_slab = slab_relaxed.get_potential_energy()

        ads_symbols = [
            s for s, t in zip(atoms.get_chemical_symbols(), atoms.get_tags())
            if t == 2
        ]
        unknown = [s for s in ads_symbols if s not in ATOMIC_REFERENCE_ENERGIES]
        if unknown:
            logging.warning(
                f"Skipping binding energy (unknown reference energies for {set(unknown)})"
            )
            return atoms_relaxed, None

        E_ads_ref = sum(ATOMIC_REFERENCE_ENERGIES[s] for s in ads_symbols)
        E_ads = E_slab_ads - E_slab - E_ads_ref
        return atoms_relaxed, {
            "slab_ads_energy": E_slab_ads,
            "slab_energy": E_slab,
            "ads_ref_energy": E_ads_ref,
            "ads_energy": E_ads,
        }
    except Exception as e:
        logging.warning(f"calc_binding_energy failed: {e}")
        return atoms_relaxed, None
