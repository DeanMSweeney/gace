
import numpy as np 
import contextlib
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
          
          if cell.dim() == 2: 
                cell = cell.unsqueeze(0) # for single graphs batch_size=1

          A =  cell[batch] # (N,3,3)
          x0 = pos # (N,3)
          x1 = pos_relaxed # (N,3)

          with ctx: 

            f0 = torch.linalg.solve(A.transpose(1,2), x0.unsqueeze(-1)).squeeze(-1)  # (N,3)
            f1 = torch.linalg.solve(A.transpose(1,2), x1.unsqueeze(-1)).squeeze(-1)  # (N,3)

            ds_f = f1 - f0 # N, 3
            ds_f -= torch.floor(ds_f + 0.5)
            ds = torch.bmm(ds_f.unsqueeze(1), A).squeeze(1)  # (N,3)

            f0 = torch.linalg.solve(A.transpose(1,2), x0.unsqueeze(-1)).squeeze(-1)
            x_recon = torch.bmm(f0.unsqueeze(1), A).squeeze(1)
            err = (x0 - x_recon).abs().max()  
            
      print(err)      
                
      return ds

    def path(
                self, 
                input_data: Data,
                from_noise: Optional[bool]=None,
    ) -> Tuple[Data, Tensor]:
        
        data = input_data.clone()
        from_noise = self.from_noise if from_noise is None else from_noise

        x1 = data.pos_relaxed # sample relaxed positions and discard

        if from_noise:
                x0 = torch.randn_like(x1) # sample noise 
        else:
                x0 = data.pos.clone() # sample initial positions and discard 

        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)

        displacement = self.pbc_aware_disp(
              pos=x0, 
              pos_relaxed=x1,
              cell=data.cell,
              batch=data.batch,
        )

        xt = x0 + t * displacement # x1 * t + (1-t) * x0
        data.pos = xt # reassign xt to pos 
        data.pos_init = x0
        data.t = t

        u = displacement

        return data, u 