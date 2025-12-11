    
import numpy as np 
import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple


def pbc_aware_delta(
            cell: Tensor, 
            delta: Tensor, 
            batch: Tensor, 
            naive: bool=True,
            norm: bool=False,
) -> Tensor:
        
        A  =  cell[batch]  # (N,3,3)
        G = A @ A.transpose(1,2) # (N, 3, 3)
        N = len(batch)
        
        u = delta # (N,3)
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
        
        if norm:
            return u_min.abs().mean(dim=1, keepdim=True) # N,3 -> N,1
        else: 
            return u_min