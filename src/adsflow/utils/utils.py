import numpy as np 
import ase
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple
import ase 
from ase.io import read, write

  
def ship_data(
    input_data: Data,
    mask: torch.Tensor,
    idx: Optional[int],
)
                
        x0 = data.pos_init[mask]
        x1 = data.pos_relaxed[mask]
        xt = data.pos[mask]
        atomic_numbers = data.atomic_numbers[mask]
        cell = data.cell[smpl]
        
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
    
        print('>>> Writing POSCARs')
        write(f"written_strucs/POSCAR0_{idx}",  x0, direct=False)  # Cartesian
        write(f"written_strucs/POSCAR1_{idx}",  x1, direct=False)  # Cartesian
        write(f"written_strucs/POSCART_{idx}",  xt, direct=False)  # Cartesian

      
    
    