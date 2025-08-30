
import numpy as np 
import itertools
import ase
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple


class OptTransportMixin: 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
    
    @staticmethod
    def masked_loss(
          u_hat: Tensor, # N, 3
          u: Tensor, # N, 3
          w: Tensor, # N, 1
    ):
    
         per_atom = ((u - u_hat) ** 2).mean(dim=1).unsqueeze(1) # N, 1
         masked_per_atom = per_atom * w # N, 1
         masked_loss = masked_per_atom.sum() / w.sum().clamp_min(1e-8) 
         
         return masked_loss
   
    @staticmethod # define as normal function within class's namespace
    def pbc_aware_disp(
                cell: Tensor, 
                pos: Tensor, 
                pos_relaxed: Tensor, 
                batch: Tensor, 
    ) -> Tensor:
          
          A  =  cell[batch]  # (N,3,3)
          G = A @ A.transpose(1,2) # (N, 3, 3)
          x0 = pos # (N,3)
          x1 = pos_relaxed # (N,3)
          N = len(batch)
          
          u = x1 - x0 # (N,3)
          u_f = torch.linalg.solve(A.transpose(1,2), u.unsqueeze(-1)).squeeze(-1)  # (N,3)
          n0 = torch.round(u_f)
          
          naive = True
          if naive:
                u_f_min = u_f - n0
                
                if u_f_min.abs().max() > 0.50: 
                      print("FLAGGED")
          else: 
                # nearest neighbor search algorithm 
                D = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), dtype=torch.int) # (27, 3)
                n = n0.unsqueeze(1) + D.unsqueeze(0) # (N, 1, 3) + (1, 27, 3) = (N, 27, 3) 
                U_f_n = u_f.unsqueeze(1) - n # (N, 1, 3) - (N, 27, 3) = (N, 27, 3)
                W = U_f_n @ G # (N, 27, 3)
                u_f2 = (W * U_f_n).sum(dim=-1) # (N, 27)
                  
                # output 
                k_min = u_f2.argmin(dim=1) # N
                u_f_min = U_f_n[torch.arange(N), k_min]
                
          u_min = torch.bmm(u_f_min.unsqueeze(1), A).squeeze(1) 
          return u_min
      

    def path(
                self, 
                input_data: Data,
                from_noise: Optional[bool]=None,
                otf_velocity: bool=False,
    ) -> Tuple[Data, Tensor]:
        
        data = input_data.clone()

        x1 = data.pos_relaxed # sample relaxed positions and discard

        if from_noise:
                x0 = torch.randn_like(x1) # sample noise 
        else:
                x0 = data.pos.clone() # sample initial positions and discard 

        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)
        
        if otf_velocity:
            u = self.pbc_aware_disp(
                  pos=x0, 
                  pos_relaxed=x1,
                  cell=data.cell,
                  batch=data.batch,
            )
        else: 
              # load data 
              u = x1 - x0

        xt = x0 + t * u # x1 * t + (1-t) * x0
        data.pos = xt # reassign xt to pos 
        data.pos_init = x0
        data.t = t

        return data, u 

if __name__ == "__main__":
      from ase import Atoms
      from ase.io import read, write
      import os
      import numpy as np
      import torch
      
            
      pos = read("POSCAR", format='vasp')
      con = read("CONTCAR", format='vasp')
      
      x0 = torch.tensor(pos.get_positions())
      x1 = torch.tensor(con.get_positions())
      cell = torch.tensor(pos.cell.array)
      
      u = OptTransportMixin.pbc_aware_disp(
            cell = cell.unsqueeze(0),
            pos = x0, 
            pos_relaxed = x1, 
            batch = torch.zeros(len(x0), dtype=torch.long),
      )
      
      xt = x0 + u
      atoms = Atoms(
                  positions=xt.detach().cpu().numpy(), # initial
                  numbers=pos.get_atomic_numbers(),
                  cell=cell.detach().cpu().numpy(),
                  pbc=True,  # or (True,True,False) for a slab
            )
      
      atoms.wrap()
      
      write("POSCART",  atoms, vasp5=True, direct=False)  # Cartesian
            