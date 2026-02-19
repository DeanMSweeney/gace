from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

import os
import shutil
import subprocess

from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE
from fairchem.core import OCPCalculator


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
