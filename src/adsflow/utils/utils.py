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


def export_positions(data: Data):
    atomic_numbers = data.atomic_numbers
    pos = data.pos 
    pos_generated = data.pos_generated
    
    if hasattr(data, "pos_realxed"):
        pos_relaxed = data.pos_relaxed

def pbc_aware_disp(
            cell: Tensor, 
            pos: Tensor, 
            pos_relaxed: Tensor, 
            batch: Tensor, 
) -> Tensor:
        
        A  =  cell[batch]  # (N,3,3)
        x0 = pos # (N,3)
        x1 = pos_relaxed # (N,3)
        
        ds = x1 - x0 # (N,3)
    
        ds_f = torch.linalg.solve(A.transpose(1,2), ds.unsqueeze(-1)).squeeze(-1)  # (N,3)
        
        ds_f -= torch.floor(ds_f + 0.5)
        d_mic = torch.bmm(ds_f.unsqueeze(1), A).squeeze(1)
        return d_mic 
      
    
    