
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

import os
import shutil
import subprocess

class OptTransportUtils(): 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
    
    def masked_loss(
        self,
        a: Tensor,  # N, 3
        b: Tensor,  # N, 3
        mask: Tensor=None,  # N, 1 -> e.g., [0, 0, 1, 1, ..., 0]
    ):
        L_per_atom = ((a - b) ** 2).mean(dim=1, keepdim=True)  # N, 1
        masked_L_per_atom = L_per_atom * mask  # N, 1
        
        # Compute masked mean loss
        masked_avg_L = masked_L_per_atom.sum() / mask.sum().clamp_min(1e-8)
        
        # Compute max (absolute) loss for stability monitoring
        max_L = masked_L_per_atom.max()
        max_L_idx = torch.argmax(masked_L_per_atom)

        # Compute masked median loss (only over nonzero mask elements)
        masked_values = L_per_atom[mask.bool()].flatten()
        masked_med_L = masked_values.median()
       
        return masked_avg_L, masked_med_L, max_L, max_L_idx
            
    def pbc_aware_delta(
                self,
                cell: Tensor, 
                delta: Tensor, 
                batch: Tensor, 
                naive: bool=True,
                avg_xyz: bool=False,
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
          
          if avg_xyz:
              return u_min.abs().mean(dim=1, keepdim=True) # N,3 -> N,1
          else: 
              return u_min
     
    def path(
        self,
        input_data: Data,
        from_noise: Optional[bool]=None,
        variant: Optional[str] = None,
        naive: str=True,
    ) -> Tuple[Data, Tensor]:
        
        data = input_data.clone()

        x1 = data.pos_relaxed.clone() # sample relaxed positions x1 ~ p(x) (final)

        if variant == "Forge":
                x0 = torch.randn_like(x1) # sample noise 
        else:
                x0 = data.pos.clone() # sample initial positions and discard x0 ~ q(x) (initial)

        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)
    
        if hasattr(data, "velocity"):
              # load from data object
              u = data.velocity
              
        else: 
               delta_xyz = x1 - x0
               u = self.pbc_aware_delta(
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
                    ase.io.write(self.input_file, atoms)  # Cartesian
        
                if self.run_vasp: 
                    self.DFT(data)
            
        # Save to current directory
        if hasattr(data, "pos_relaxed"):
            
            e_gen = self.pbc_aware_delta(
                delta=data.pos_generated - data.pos_relaxed, # write generate possitional coordinates to pos_gen attribute 
                cell=data.cell,
                batch=data.batch,
        )
            
            e_null = self.pbc_aware_delta(
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
        
    def analyze_loss(
        self, 
        data: Data,
        loss: Tensor, 
        median_loss: Tensor, 
        max_loss: Tensor, 
        max_loss_idx: Tensor, 
        sample_x1: bool=False,
    ): 
        # activate loop if loss is greater than threshold 
            if loss > 0.0010: 
                self.bad_iter += 1
                
                """ 
                Compute statistics for max velocity field loss 
                """
                max_loss_z = data.atomic_numbers[max_loss_idx] # atomic number of atom with max loss
                max_loss_tag = data.tags[max_loss_idx] # tag of atom with max loss 
                
                # evaluate distance from adsorbate for surface atoms 
                if float(max_loss_tag) != 2:
                    max_atom_pos = data.pos_relaxed[max_loss_idx] #1,3
                    adsorbate = (data.tags == 2) 
                    adsorbate_pos = data.pos_relaxed[adsorbate] #N_a, 3, N_a -> # of adsorbate atoms 
                    
                    # pbc corrected distance calc 
                    K = (max_atom_pos - adsorbate_pos).size(0)
                    delta_max_ads = self.pbc_aware_delta(
                    delta=max_atom_pos - adsorbate_pos,
                    cell=data.cell,
                    batch=torch.zeros(K, dtype=torch.long),
                    naive=True,
                    avg_xyz=True,
                    )
                    # collect diagnostics for surface atom 
                    self.DS0.append(float(delta_max_ads.min())) # distance between atom and adsorbate
                    
                """
                Compute statistics for max distance moved during relaxation 
                """
                # pbc aware displacement from relaxation 
                delta = self.pbc_aware_delta(
                    delta=data.pos - data.pos_relaxed,
                    cell=data.cell,
                    batch=data.batch,
                    naive=True,
                    avg_xyz=True,
                ) 
                
                # N,1, zero-out frozen atoms and average
                distance_per_atom = delta * self.free_atoms  
                avg_distance = distance_per_atom.sum() / self.free_atoms.sum().clamp_min(1e-8)
                max_distance = distance_per_atom.max()
                max_distance_idx = torch.argmax(distance_per_atom)
                distance_of_max_loss = distance_per_atom[max_loss_idx]

                max_distance_z = data.atomic_numbers[max_distance_idx] # atomic number of atom with max distance moved 
                max_distance_tag = data.tags[max_distance_idx] # tag of atom with max distance moved 
                
                if sample_x1: 
                    # generate structures and write files for problematic losses
                    self.output_file = f"strucs/BAD-output{self.bad_iter}_{loss}_{max_loss}_{max_loss_idx}_{max_loss_z}_{max_loss_tag}.vasp"
                    self.input_file = f"strucs/BAD-input{self.bad_iter}.vasp"
                    self.relaxed_file = f"strucs/BAD-relaxed{self.bad_iter}.vasp"
                    
                    # define time interval for the IVP
                    t = torch.linspace(0., 1., 100, device=data.pos.device, dtype=data.pos.dtype)
                    opt_graph, struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse = self.sampler(data, t=t)
            
                # print statements 
                print(f"High velocity field loss {loss} ❌ || total count: {self.bad_iter}/{self.iter}")
                
                # collect diagnostics 
                self.DS1.append(int(max_loss_z)) # atomic number 
                self.DS2.append(int(max_loss_tag)) # tag (surface, bulk, adsorbate)
                self.DS3.append(int(max_distance_z)) # atomic number 
                self.DS4.append(int(max_distance_tag)) # tag (surface, bulk, adsorbate)
                self.DS5.append(float(max_distance)) # max distance moved during relaxation 
                self.DS6.append(float(distance_of_max_loss)) # max distance moved of atom with max loss 
                self.DS7.append(float(avg_distance)) # average distance moved of all atoms in struc 
        
    def DFT(
        self,
        data: Data,
    ):
        # Get the current working directory to return to later
        original_dir = os.getcwd()

        try:
            # Get the sid from the data object
            sid = data.sid

            # Create the strucs directory if it doesn't exist
            if not os.path.exists("./strucs"):
                os.makedirs("./strucs")

            # Create the sid directory within strucs
            sid_dir = os.path.join("./strucs", str(sid))
            os.makedirs(sid_dir, exist_ok=True)
            
            # Function to build simulation directory
            def build_vasp_dir(sim_dir: Optional[bool]=None, struc_file: Optional[bool]=None):
                sim_path = os.path.join(sid_dir, sim_dir)
                os.makedirs(sim_path, exist_ok=True)
                shutil.move(struc_file, os.path.join(sim_path, "POSCAR"))
                
                for file in ["INCAR", "KPOINTS", "VASP.sh"]:
                    if os.path.exists(file):
                        shutil.copy(file, sim_path)
                    else: 
                        raise AttributeError(f"Missing {file} file for VASP configuration.")

                    # Submit the .sh file in each directory
                    os.chdir(sim_path)
                    subprocess.run(["potcar.sh"])
                    subprocess.run(["sbatch", "VASP.sh"])
                    os.chdir(original_dir)
                
            # Move and rename files to their respective directories
            if os.path.exists(self.output_file):
                build_vasp_dir(sim_dir="generated", struc_file=self.output_file)
            else: 
                raise AttributeError("Generated strucutre file does not exist")
            
            if self.mode == "eval":
                if os.path.exists(self.input_file):
                    build_vasp_dir(sim_dir="inputted", struc_file=self.input_file)
                    
                if os.path.exists(self.relaxed_file) and self.mode == "eval":
                    build_vasp_dir(sim_dir="relaxed", struc_file=self.relaxed_file)
                
        finally:
            # Always return to the original directory
            os.chdir(original_dir)