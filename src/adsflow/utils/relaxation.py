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

from adsflow.utils.atoms import data_to_atoms

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


def relax_struc(
    atoms: Atoms,
    run_mlff: bool = False,
    config_vasp: bool = False,
    mlff_checkpoint: Optional[str] = None,
    accelerator: str = "gpu",
    seed: int = 42,
) -> Data:
    """
    Post-generation relaxation step.

    Applies either MLFF relaxation or configures DFT input files depending
    on which option is enabled. If neither is set, returns opt_graph unchanged.

    Args:
        opt_graph: Graph with pos_generated set to the flow-matched positions
        run_mlff: Whether to run MLFF relaxation
        config_vasp: Whether to configure VASP DFT input files
        mlff_checkpoint: MLFF model checkpoint name (required if run_mlff=True)
        accelerator: Device type ("gpu" or "cpu") for OCPCalculator
        seed: Random seed for OCPCalculator
        gen_POS: Path to generated structure file (required if config_vasp=True)
        in_POS: Path to initial structure file
        rlxd_POS: Path to relaxed structure file

    Returns:
        opt_graph with pos_generated updated (MLFF) or DFT inputs written (config_vasp)
    """
    if run_mlff:
        with Silent():
            calculator = OCPCalculator(
                model_name=mlff_checkpoint,
                local_cache="./mlff_checkpoint_cache",
                cpu=(accelerator == "cpu"),
                seed=seed,
            )
            atoms_relaxed, _, _ = relax_w_mlff(
                atoms=atoms,
                calculator=calculator,
                fmax=0.05,
                steps=200,
                optimizer="BFGS",
            )
    else:
        return None

    return atoms_relaxed
