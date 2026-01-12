    
import numpy as np 
import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple
import itertools
#from torch_linear_assignment import assignment_to_indices


def pbc_aware_delta(
            cell: Tensor, 
            delta: Tensor, 
            batch: Optional[Tensor]=None, 
            naive: bool=True,
            norm: bool=False,
) -> Tensor:
    
        """
            Cell either (batch_size, 3, 3) from batched trainign data or (3, 3) from input 
            Cell/A must be converted to (N, 3, 3) for parallelization algorithm to work 
        """
        
        if not isinstance(cell, Tensor):
            cell = torch.from_numpy(np.asarray(cell)).to(dtype=torch.float32)
        if not isinstance(delta, Tensor):
            delta = torch.from_numpy(np.asarray(delta)).to(dtype=torch.float32)
        
        N = torch.zeros(delta.size(0), dtype=torch.long) # number of atoms 
        A = cell.unsqueeze(0)[N] if batch is None else cell[batch] 
    
        u = delta # (N,3)
        u_f = torch.linalg.solve(A.transpose(1,2), u.unsqueeze(-1)).squeeze(-1)  # (N,3)
        n0 = torch.round(u_f)
        
        if naive:
            u_f_min = u_f - n0
        else: 
            # nearest neighbor search algorithm 
            G = A @ A.transpose(1,2) # (N, 3, 3)
            D = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), dtype=torch.int, device=n0.device) # (27, 3)
            n = n0.unsqueeze(1) + D.unsqueeze(0) # (N, 1, 3) + (1, 27, 3) = (N, 27, 3) 
            U_f_n = u_f.unsqueeze(1) - n # (N, 1, 3) - (N, 27, 3) = (N, 27, 3)
            W = U_f_n @ G # (N, 27, 3)
            u_f2 = (W * U_f_n).sum(dim=-1) # (N, 27)
                
            # output 
            k_min = u_f2.argmin(dim=1) # N
            u_f_min = U_f_n[torch.arange(delta.size(0)), k_min]
            
        u_min = torch.bmm(u_f_min.unsqueeze(1), A).squeeze(1)
        
        if norm:
            return u_min.abs().mean(dim=1, keepdim=True) # N,3 -> N,1
        else: 
            return u_min
"""
def hungarian(
    s0: Tensor,
    s1: Tensor,
    cell: Tensor,
    batch: Tensor, 
) -> Tensor: 
 
    # compute pairwise cost matrix for atomic positions with minimum image convention 
    C = s1.unsqueeze(0) - s0.unsqueeze(1) # (N, N, 3)
    C = pbc_aware_delta(
        cell=cell, 
        delta=C.reshape(-1, 3), #(N*N, 3)
        batch=batch,
    ).reshape(s1.size(0), s1.size(0), 3) # N, N, 3
    
    C_l2 = torch.linalg.norm(C, dim=-1)

    _, col_ind = assignment_to_indices(cost=C_l2)
    s0_aligned = s0[col_ind]
    
    return s0_aligned

"""      
