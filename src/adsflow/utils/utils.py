import os
import itertools

import numpy as np 
from ase import Atoms
from ase.io import read, write

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple

from scipy import linear_sum_assignment


def hungarian_horntail(
    x0: Tensor,
    x1: Tensor,
    mask: Tensor,
    cell: Tensor,
    naive: bool=True,
) -> Tensor: 
 
    # compute pairwise cost matrix for atomic positions with minimum image convention 
    x0 = x0[mask]
    x1 = x1[mask]
    Cx = x1.unsqueeze(0) - x0.unsqueeze(1) # (N, N, 3)
    Cx = minimum_image_distance(
        cell=cell, 
        delta=Cx.reshape(-1, 3), #(N*N, 3)
        naive=naive,
    )
    Cx = (Cx.reshape(x1.size(0), x1.size(0), 3))**2 # N, N, 3
    Cx_l2 = Cx.sum(dim=-1) # N, N

    # Hungarian algorithm for atomic postion assignment problem 
    x0 = x0.detach().cpu().numpy() # N, 3
    _, col_ind = linear_sum_assignment(cost=Cx_l2.detach().cpu().numpy() , maximize=False)
    x0_aligned = torch.tensor(x0[col_ind], dtype=Cx.dtype, device=Cx.device)
    
    return x0_aligned
        
    """
    # compute pairwise cost matrix for atom types 
    z0 = data.atomic_numbers[mask]
    z1 = data.true_atomic_numbers[mask]
    Cz = torch.cdist(z1, z0, p=2)
    
    # Hyngarian algorithm for atomic postion assignment problem 
    z0 = z0.detach().cpu().numpy() 
    _, col_ind = linear_sum_assignment(cost=Cz.detach().cpu().numpy() , maximize=False)
    data.atomic_numbers[mask] = torch.tensor(z0[col_ind], dtype=Cz.dtype, device=Cz.device)
    """   
def minimum_image_distance(
            cell: Tensor, 
            delta: Tensor, # N, 3
            batch: Optional[Tensor]=None, 
            naive: bool=True,
) -> Tensor:
        
        if batch is not None: 
            A  =  cell[batch]  # (N,3,3) - repeate unit cell for all atoms in a graph
        else: 
            A = cell.expand(delta.size(0), 3, 3) # (3,3) -> (N, 3, 3)
            
        delta_f = torch.linalg.solve(A.transpose(1,2), delta.unsqueeze(-1)).squeeze(-1)  # (N,3)
        n0 = torch.round(delta_f)
        
        if naive:
            delta_f_min = delta_f - n0
        else: 
            # nearest neighbor search algorithm 
            G = A @ A.transpose(1,2) # (N, 3, 3)
            N = delta.size(0)
            D = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), dtype=torch.int, device=n0.device) # (27, 3)
            n = n0.unsqueeze(1) + D.unsqueeze(0) # (N, 1, 3) + (1, 27, 3) = (N, 27, 3) 
            delta_f_n = delta_f.unsqueeze(1) - n # (N, 1, 3) - (N, 27, 3) = (N, 27, 3)
            W = delta_f_n @ G # (N, 27, 3)
            delta_f2 = (W * delta_f_n).sum(dim=-1) # (N, 27)
                
            # output 
            k_min = delta_f2.argmin(dim=1) # N
            delta_f_min = delta_f_n[torch.arange(N), k_min]
            
        delta_min = torch.bmm(delta_f_min.unsqueeze(1), A).squeeze(1)
            
        return delta_min
    
    
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

      
    
    