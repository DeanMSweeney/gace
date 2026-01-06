import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from adsflow.utils.post_processing import config_DFT, write_strucs
from adsflow.utils.invariance import pbc_aware_delta
from adsflow.utils.eval_metrics import ADwT


class OptTransport(): 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
    
    def masked_loss(
        self,
        input_data,
        a: Tensor,  # N, 3
        b: Tensor,  # N, 3
        alpha: float=0, # masked MSE
        tune_adsorbate: bool=False,
    ):  
        # compute per atom loss
        with torch.no_grad():
            data = input_data
            mask = (1 - data.fixed).unsqueeze(1) # # N, 1 -> e.g., [0, 0, ..., 0, 1, ..., 1] (0 <-> frozen, 1 <-> free)
            # compute weights with numerical stability
            if alpha == 0: # weight all samples equally 
                w = 1 / mask.sum().clamp_min(1e-8) # equal weighting, equivalent to L = 1/N * sum(l_per_atom), w = 1/N
            else:
                if tune_adsorbate: 
                    alpha = 4 if alpha == 1 else alpha 
                    w0 = torch.ones_like(mask) * (1 / mask.sum().clamp_min(1e-8)) # N,1 
                    ads_weight = (data.tags == 2).float().unsqueeze(1) # N,1 -> [0, 0, 0, 0, 1, ..., 1]
                    w =  (w0 * (1 + ads_weight * (alpha - 1)))  # N,1 
                    w = w / w.sum()
                else:
                    # compute distance moved by all atoms
                    if hasattr(data, "velocity"):
                        m = mask.squeeze(-1).bool()
                        d = torch.linalg.norm(data.velocity, dim=1, keepdim=True) #N,3 -> N,1
                        d_sum_per_graph = scatter(src=d[m], index=data.batch[m], dim=0, reduce="mean")  # num_graphs, 1
                        d = d / d_sum_per_graph[data.batch] # normalize distance by average movement per graph 
                    else:
                        d = pbc_aware_delta(
                            cell=data.cell,
                            delta=(data.pos_init - data.pos_relaxed),
                            batch=data.batch,
                            norm=True, # average distance for each atom
                        )
                    d_alpha = d ** alpha
                    den = scatter(src=d_alpha, index=data.batch, dim=0, reduce="sum")  # num_graphs, 1
                    w = d_alpha / den[data.batch].clamp_min(1e-8)  # N,1 - weights per graph not batch 

        # compute masked weighted loss (sum of weighted per-atom losses)
        L_per_atom = ((a - b) ** 2).mean(dim=1, keepdim=True)  # N, 1
        masked_L_per_atom = L_per_atom * mask  # N, 1
        weighted_L_per_atom = w * masked_L_per_atom
        L = weighted_L_per_atom.sum()
            
        with torch.no_grad():
            # Compute max (absolute) loss for stability monitoring
            max_L = masked_L_per_atom.max()
            max_L_idx = torch.argmax(masked_L_per_atom)

        return L, max_L, max_L_idx
            
    def path(
        self,
        input_data: Data,
        from_noise: Optional[bool]=None,
        variant: Optional[str] = None,
        naive: str=True,
    ) -> Tuple[Data, Tensor]:
        
        with torch.no_grad():
            data = input_data.clone()

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
  
    def sample(
            self, 
            input_data: Data,
            t: torch.Tensor,
            threshold: float=0.20,
            
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
        frozen_atoms = (data.fixed == 1)
        data.pos_generated[frozen_atoms] = data.pos[frozen_atoms]
        
        if self.gen_POS:
                write_strucs(
                    data=data, 
                    gen_POS=self.gen_POS, 
                    in_POS=self.in_POS, 
                    rlxd_POS=self.rlxd_POS,
                )
                
                if self.config_vasp: 
                    config_DFT(
                        data=data, 
                        gen_POS=self.gen_POS,
                        in_POS=self.in_POS,  
                        rlxd_POS=self.rlxd_POS, 
                    )
        else: 
            print("Output file not provided... structure will not be saved")
                
        # Save to current directory
        if hasattr(data, "pos_relaxed"):
            
            # collect data for analysis
            mask = (1 - data.fixed).bool()
            cell = data.cell 
            batch = data.batch 
            pos_init = data.pos 
            pos_predicted = data.pos_generated
            pos_target = data.pos_relaxed 
            
            generated_mae, generated_dwt, generated_adwt = ADwT(
               pos_predicted=pos_predicted, 
               pos_target=pos_target,
               threshold=threshold,
               fixed_atoms_mask=mask,
               cell=cell, 
               batch=batch, 
           )
            
            initial_mae, initial_dwt, initial_adwt = ADwT(
               pos_predicted=pos_init, 
               pos_target=pos_target,
               threshold=threshold,
               fixed_atoms_mask=mask,
               cell=cell, 
               batch=batch, 
           )
            return data, generated_mae, generated_dwt, generated_adwt, initial_mae, initial_dwt, initial_adwt
        else: 
            return data
  