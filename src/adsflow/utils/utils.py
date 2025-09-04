import os

import numpy as np 
from ase import Atoms
from ase.io import read, write

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple

def ship_data(
    input_data: Data,
    sample: torch.Tensor,
    idx: Optional[int],
):

    dir_name = "shipped_data" # directory to store data       
    data = input_data.clone()
    
    # build mask for selected atoms and output atomic positions 
    mask = (data.batch == sample)
    x0 = data.pos_init[mask]
    x1 = data.pos_relaxed[mask]
    xt = data.pos[mask]
    atomic_numbers = data.atomic_numbers[mask]
    cell = data.cell[sample]
    
    # convert atomic positions to ASE objects 
    x0 = Atoms(
            positions=x0.detach().cpu().numpy(), # initial
            numbers=atomic_numbers.detach().cpu().numpy(),
            cell=cell.squeeze(0).detach().cpu().numpy(),
            pbc=True,  # or (True,True,False) for a slab
        )
    x1 = Atoms(
            positions=x1.detach().cpu().numpy(), # initial
            numbers=atomic_numbers.detach().cpu().numpy(),
            cell=cell.squeeze(0).detach().cpu().numpy(),
            pbc=True,  # or (True,True,False) for a slab
        )
    xt = Atoms(
            positions=xt.detach().cpu().numpy(), # initial
            numbers=atomic_numbers.detach().cpu().numpy(),
            cell=cell.squeeze(0).detach().cpu().numpy(),
            pbc=True,  # or (True,True,False) for a slab
        )

    # write to POSCARs
    print('>>> Writing POSCARs 👩‍💻')
    os.makedirs(dir_name, exist_ok=True)
    write(f"{dir_name}/POSCAR0_{idx}",  x0, direct=False)  # Cartesian
    write(f"{dir_name}/POSCAR1_{idx}",  x1, direct=False)  # Cartesian
    write(f"{dir_name}/POSCART_{idx}",  xt, direct=False)  # Cartesian
    
    print(f'Wrote POSCARS to ./{dir_name}')

      
    
    