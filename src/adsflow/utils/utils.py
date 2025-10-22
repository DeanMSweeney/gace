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
    data: torch.Tensor,
    naive: bool=True,
): 
    catalyst = (data.tags != 2) # mask adsorbate atoms
    num_samples = int(data.num_graphs)
    for idx in range(num_samples):
        
        sample = data.batch == idx
        mask = sample & catalyst
        
        x0 = data.pos[mask]
        z0 = data.z[mask]
    
        x1 = data.pos_relaxed[mask]
        z1 = data.atomic_numbers[mask]
        
        # compute cost matrix for atomic positions with minimum image convention 
        Cx_raw = x1[None, :, :] - x0[:,  None, :]
        Cx = minimum_image_distance(
            cell=data.cell[idx], 
            cost=Cx_raw, 
            naive=naive,
        )
        Cx = Cx**2
        
        # compute cost matrix for atomic numbers 
        Cz = torch.cdist(z1, z0, p=2)**2
        Cz = Cz**2
        
        # Hyngarian algorithm for atomic postion assignment problem 
        x0 = data.pos[mask].detach().cpu().numpy() # N, 3
        _, col_ind = linear_sum_assignment(cost=Cx.detach().cpu().numpy() , maximize=False)
        data.pos[mask] = torch.tensor(x0[col_ind], dtype=Cx.dtype, device=Cx.device)
        
        # Hungarian algorithm for atom types assignment problem 
        z0 = data.z.detach().cpu().numpy() # N, 3
        c_z = Cz.detach().cpu().numpy() # N, 3
        _, col_ind = linear_sum_assignment(cost=c_z, maximize=False)
        data.z[mask] = torch.tensor(z0[col_ind], dtype=Cz.dtype, device=Cz.device)
           
def minimum_image_distance(
            cell: Tensor, 
            distance: Tensor, 
            batch: Optional[Tensor]=None, 
            naive: bool=True,
) -> Tensor:
        
        if batch is None: 
            batch = torch.ones(distance.size(0))
            
        A  =  cell[batch]  # (N,3,3)
        G = A @ A.transpose(1,2) # (N, 3, 3)
        N = len(batch)
        
        u = distance # (N,3)
        u_f = torch.linalg.solve(A.transpose(1,2), u.unsqueeze(-1)).squeeze(-1)  # (N,3)
        n0 = torch.round(u_f)
        
        if naive:
            u_f_min = u_f - n0
        else: 
            # nearest neighbor search algorithm 
            D = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), dtype=torch.int, device=n0.device) # (27, 3)
            n = n0.unsqueeze(1) + D.unsqueeze(0) # (N, 1, 3) + (1, 27, 3) = (N, 27, 3) 
            U_f_n = u_f.unsqueeze(1) - n # (N, 1, 3) - (N, 27, 3) = (N, 27, 3)
            W = U_f_n @ G # (N, 27, 3)
            u_f2 = (W * U_f_n).sum(dim=-1) # (N, 27)
                
            # output 
            k_min = u_f2.argmin(dim=1) # N
            u_f_min = U_f_n[torch.arange(N), k_min]
            
        u_min = torch.bmm(u_f_min.unsqueeze(1), A).squeeze(1)
            
        return u_min
    
    
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

      
    
    