
import numpy as np 
import itertools
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

from adsflow.utils.post_processing import config_DFT
from adsflow.utils.invariance import pbc_aware_delta

import os
import shutil
import subprocess

class OptTransport(): 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
    
    def masked_loss(
        self,
        input_data,
        a: Tensor,  # N, 3
        b: Tensor,  # N, 3
        mask: Tensor=None,  # N, 1 -> e.g., [0, 0, 1, 1, ..., 0]
    ):
        # compute per atom loss 
        L_per_atom = ((a - b) ** 2).mean(dim=1, keepdim=True)  # N, 1
        masked_L_per_atom = L_per_atom * mask  # N, 1
        data = input_data
        
        weight_by_dist = False
        target_adsorbate = False
        if weight_by_dist:
            # compute distance moved by all atoms 
            d = pbc_aware_delta(
                cell=data.cell, 
                delta=(data.pos_init - data.pos_relaxed),
                batch=data.batch, 
                avg_xyz=True, # average distance for each atom 
            )
            
            # compute normalized displacement and weights 
            r = d / d.sum()
            w = r / (r.sum()) 
            
            # compute masked loss 
            L = w * masked_L_per_atom.sum()
                
        elif target_adsorbate:
            pass
        
        else: # mask frozen atoms only
            
            # Compute masked mean loss
            L = masked_L_per_atom.sum() / mask.sum().clamp_min(1e-8)
            
            
        # Compute max (absolute) loss for stability monitoring
        max_L = masked_L_per_atom.max()
        max_L_idx = torch.argmax(masked_L_per_atom)

        # Compute masked median loss (only over nonzero mask elements)
        masked_values = L_per_atom[mask.bool()].flatten()
        median_L = masked_values.median()
       
        return L, median_L, max_L, max_L_idx
            
    def path(
        self,
        input_data: Data,
        from_noise: Optional[bool]=None,
        variant: Optional[str] = None,
        naive: str=True,
    ) -> Tuple[Data, Tensor]:
        
        data = input_data

        x1 = data.pos_relaxed # sample relaxed positions x1 ~ p(x) (final)

        if variant == "Forge":
                x0 = torch.randn_like(x1) # sample noise 
        else:
                x0 = data.pos # sample initial positions x0 ~ q(x) (initial)

        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)
    
        if hasattr(data, "velocity"):
              # load from data object
              u = data.velocity
              
        else: 
               delta_xyz = x1 - x0
               u = pbc_aware_delta(
                  delta=delta_xyz,
                  cell=data.cell,
                  batch=data.batch,
                  naive=naive,
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
                  pbc=True,  # 
                )
              atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
              ase.io.write(self.output_file, atoms)  # Cartesian
              
              if self.input_file:
                # Save to current directory
                atoms = Atoms(
                    positions=x0_np, # initial
                    numbers=z_np,
                    cell=cell_np,
                    pbc=True,  # or (True, True, False) for a slab
                    )
                atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
                ase.io.write(self.input_file, atoms)  # Cartesian
                
                if self.relaxed_file:
                    # Save to current directory
                    atoms = Atoms(
                        positions=xdft_np, # initial
                        numbers=z_np,
                        cell=cell_np,
                        pbc=True,  # 
                        )
                    atoms.set_constraint(FixAtoms(mask=f_np.astype(bool)))
                    ase.io.write(self.relaxed_file, atoms)  # Cartesian
        
                if self.config_vasp: 
                    config_DFT(
                        data=data, 
                        in_POS=self.in_POS, 
                        gen_POS=self.gen_POS, 
                        rlxd_POS=self.rlxd_POS,
                        
                    )
        else: 
            print("Output file not provided... structure will not be saved")
                
        # Save to current directory
        if hasattr(data, "pos_relaxed"):
            
            e_gen = pbc_aware_delta(
                delta=data.pos_generated - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
                cell=data.cell,
                batch=data.batch,
        )
            
            e_null = pbc_aware_delta(
                delta=data.pos - data.pos_relaxed, # write generated possitional coordinates to pos_gen attribute 
                cell=data.cell,
                batch=data.batch,
        )
            
            struc_mse, struc_mdse, _, _ = self.masked_loss(a=e_gen, b=torch.zeros_like(e_gen), mask=self.free_atoms)
            struc_rmse, struc_rmdse = torch.sqrt(struc_mse), torch.sqrt(struc_mdse)
            
            null_struc_mse, null_struc_mdse, _, _ = self.masked_loss(a=e_null, b=torch.zeros_like(e_null), mask=self.free_atoms)
            null_struc_rmse, null_struc_rmdse = torch.sqrt(null_struc_mse), torch.sqrt(null_struc_mdse)
            
        
            return data, struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse
        else: 
            return data
  