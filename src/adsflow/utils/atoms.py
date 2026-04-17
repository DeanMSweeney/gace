import ase
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from adsflow.utils.invariance import pbc_aware_delta

import os
import shutil
import subprocess


def data_to_atoms(data: Data, pos_type: Optional[torch.Tensor]=None) -> Atoms:
    """
    Convert a PyTorch Geometric Data object to an ASE Atoms object.
    Assumes batch_size=1 (single structure from DataLoader).

    Args:
        data: PyG Data object with pos, atomic_numbers, cell, fixed, tags

    Returns:
        ASE Atoms object with positions, cell, and constraints
    """
    pos_type = data.pos if pos_type is None else pos_type
    pos = pos_type.cpu().numpy()
    atomic_numbers = data.atomic_numbers.cpu().numpy()
    fixed = data.fixed.cpu().numpy()
    tags = data.tags.cpu().numpy()
    cell = data.cell.squeeze(0).cpu().numpy()

    atoms = Atoms(
        positions=pos,
        numbers=atomic_numbers,
        cell=cell,
        pbc=True,
        tags=tags,
    )

    if fixed.any():
        atoms.set_constraint(FixAtoms(mask=fixed.astype(bool)))

    return atoms

def write_strucs(
    data: Data, 
    gen_POS: str, 
    in_POS: Optional[str]=None, 
    rlxd_POS: Optional[str]=None,  
):
    
    if in_POS:
        atoms = data_to_atoms(data, pos_type=data.pos)
        ase.io.write(in_POS, atoms)  # Cartesian
    
    if gen_POS:
        atoms = data_to_atoms(data, pos_type=data.pos_generated)
        ase.io.write(gen_POS, atoms)  # Cartesian
    
    if rlxd_POS:
        atoms = data_to_atoms(data, pos_type=data.pos_relaxed)
        ase.io.write(rlxd_POS, atoms)  # Cartesian
        
