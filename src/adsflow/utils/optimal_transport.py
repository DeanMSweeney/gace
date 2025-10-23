
import numpy as np 
import ase
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple
from adsflow.utils.utils import minimum_image_distance

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
          L_per_atom = ((a - b) ** 2).mean(dim=1).unsqueeze(1) # N, 1
          masked_L_per_atom = L_per_atom * mask # N, 1
          masked_L = masked_L_per_atom.sum() / mask.sum().clamp_min(1e-8)
          max_L = masked_L_per_atom.abs().max()
          
          return masked_L, max_L
                
    def path(
                self,
                input_data: Data,
                variant: Optional[str] = None,
    ) -> Tuple[Data, Tensor]:
        
        data = input_data.clone()
        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)

        if variant == "forge":
                # sample target postions, atom types
                x1 = data.pos_relaxed.clone() # x1 ~ p(x) 
                z1 = data.atomic_number.clone() # z1 ~ p(z) 
                
                # sample initial positions and atom types 
                x0 = torch.randn_like(x1) # x0 ~ q(x)
                z0 = torch.randn_like(z1) # z0 ~ q(z)
                
                # assign adsorbate atoms to correct positions and atom types 
                ads_mask = (data.tags == 2)
                x0[ads_mask] = data.pos[ads_mask]
                z0[ads_mask] = data.atomic_number[ads_mask]
                
                data.pos = x0 
                data.true_atomic_numbers = z1
                data.atomic_numbers = z0
                
        elif variant == "refine":
            x1 = data.pos_relaxed.clone() # sample relaxed positions x1 ~ p(x) 
            x0 = data.pos.clone() # sample initial positions x0 ~ q(x) 
    
            if hasattr(data, "velocity"):
                # load from data object
                u = data.velocity
    
            else: 
               u_var = x1 - x0
               u = minimum_image_distance(
                  delta=u_var,
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
      
        data.pos = input_data.pos
        data.pos_generated = xt[-1]
        mask = (data.fixed == 1)
        data.pos_generated[mask] = data.pos[mask]
        
        if self.output_file: 
              z_np = data.atomic_numbers.detach().cpu().numpy()
              cell_np = data.cell.squeeze(0).detach().cpu().numpy()
              x1_np = data.pos_generated.detach().cpu().numpy()
              x0_np = data.pos.detach().cpu().numpy()
              xdft_np = data.pos_relaxed.detach().cpu().numpy()
              f_np = data.fixed.detach().cpu().numpy()
              
              # Save to current directory
              atoms = Atoms(
                  positions=x1_np, # initial
                  numbers=z_np,
                  cell=cell_np,
                  pbc=True,  # or (True, True, False) for a slab
                )
              atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
              ase.io.write(self.output_file, atoms)  # Cartesian
              
              # Save to current directory
              atoms = Atoms(
                  positions=x0_np, # initial
                  numbers=z_np,
                  cell=cell_np,
                  pbc=True,  # or (True, True, False) for a slab
                )
              atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
              if self.input_file:
                  ase.io.write(self.input_file, atoms)  # Cartesian
              else: 
                  ase.io.write("struc_x0.vasp", atoms)
                  
              # Save to current directory
              atoms = Atoms(
                  positions=xdft_np, # initial
                  numbers=z_np,
                  cell=cell_np,
                  pbc=True,  # or (True, True, False) for a slab
                )
              atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
              ase.io.write("relaxed.vasp", atoms)  # Cartesian
              
        if hasattr(data, "pos_relaxed"):
              e_gen = minimum_image_distance(
                  delta=data.pos_generated - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
                  cell=data.cell,
                  batch=data.batch,
                  naive=self.naive,
            )
              
              e_null = minimum_image_distance(
                  delta=data.pos - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
                  cell=data.cell,
                  batch=data.batch,
                  naive=self.naive,
            )
              
              struc_mse, _ = __class__.masked_loss(a=e_gen, b=torch.zeros_like(e_gen), mask=self.free_atoms)
              struc_rmse = torch.sqrt(struc_mse)
              
              null_struc_mse, _ = __class__.masked_loss(a=e_null, b=torch.zeros_like(e_null), mask=self.free_atoms)
              null_struc_rmse = torch.sqrt(null_struc_mse)
            
              return data, struc_rmse, null_struc_rmse
        else: 
            return data