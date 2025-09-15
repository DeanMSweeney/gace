
import numpy as np 
import itertools
import ase
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple


class OptTransportMixin: 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
    
    @staticmethod
    def masked_loss(
          a: Tensor, # N, 3
          b: Tensor, # N, 3
          mask: Tensor, # N, 1 -> w =[0, 0, 1, 1, ..., 0]
    ):
          per_atom = ((a - b) ** 2).mean(dim=1).unsqueeze(1) # N, 1
          masked_per_atom = per_atom * mask # N, 1
          masked_loss = masked_per_atom.sum() / mask.sum().clamp_min(1e-8)
          
          return masked_loss
               
            
    @staticmethod # define as normal function within class's namespace
    def pbc_aware_pos(
                cell: Tensor, 
                positions: Tensor, 
                batch: Tensor, 
                naive: bool=True,
    ) -> Tensor:
          
          A  =  cell[batch]  # (N,3,3)
          G = A @ A.transpose(1,2) # (N, 3, 3)
          N = len(batch)
          
          u = positions # (N,3)
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
     
    def path(
                self,
                input_data: Data,
                from_noise: Optional[bool]=None,
                variant: Optional[str] = None,
    ) -> Tuple[Data, Tensor]:
        
        data = input_data.clone()

        x1 = data.pos_relaxed.clone() # sample relaxed positions x1 ~ p(x) (final)

        if from_noise or variant == "Forge":
                x0 = torch.randn_like(x1) # sample noise 
        else:
                x0 = data.pos.clone() # sample initial positions and discard x0 ~ q(x) (initial)

        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)
    
        if not self.otf_velocity and hasattr(data, "velocity"):
              # load from data object
              u = data.velocity
              
        else: 
               u_var = x1 - x0
               u = __class__.pbc_aware_pos(
                  positions=u_var,
                  cell=data.cell,
                  batch=data.batch,
                  naive=self.naive,
            )

        xt = x0 + t * u # x1 * t + (1-t) * x0
        data.pos = xt # reassign xt to pos 
        data.pos_init = x0
        data.t = t

        return data, u 
  
    def sampler(
            self, 
            input_data: Data,
            t: torch.Tensor,
            
    ) -> Union[Data, Tuple[Data, torch.Tensor]]:
        
        data = input_data.clone()
        def dx_dt(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            """
            Wrap Pytorch model in a function with format: f(t, y)
            """
            data.pos = xt
            data.t = t.view(1,1).expand(len(xt), -1)
            u = self(data)
            return u 
        
        xt = odeint(
            func=dx_dt, 
            y0=input_data.pos,
            t=t,
            rtol=self.rtol,
            atol=self.atol,
            method=self.ode_method,
        )
        
        data.pos_generated = xt[-1]
        
        mask = [data.fixed ==1]
        data.pos_generated[mask] = data.pos[mask]

        data.pos = input_data.pos
        
        if hasattr(data, "pos_relaxed"):
              e_gen = __class__.pbc_aware_pos(
                  positions=data.pos_generated - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
                  cell=data.cell,
                  batch=data.batch,
                  naive=self.naive,
            )
              e_null = __class__.pbc_aware_pos(
                  positions=data.pos - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
                  cell=data.cell,
                  batch=data.batch,
                  naive=self.naive,
            )

            struc_rmse = torch.sqrt(__class__.masked_loss(a=e_gen, b=torch.zeros_like(e_gen), mask=self.free_atoms))
            null_struc_rmse = torch.sqrt(__class__.masked_loss(a=e_null, b=torch.zeros_like(e_null), mask=self.free_atoms))
            return data, struc_rmse, null_struc_rmse
      
        else: 
            return data