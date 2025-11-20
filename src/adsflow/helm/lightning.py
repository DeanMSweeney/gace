import os
import numpy as np
import torch
from torch import optim 
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, PolynomialLR
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch_geometric.data import Data
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from adsflow.utils.flow_matching import OptTransportMixin
from adsflow.utils.utils import ship_data

from typing import Optional, Literal, Union, Tuple


class LightFlowMatch(LightningModule, OptTransportMixin):
    """
    This class defines the training algorithm for the conditional flow matching model using optimal transport 
    
    Args
    ---------
    model: dict 
        Model configuration 
    lr: int
        Base learning rate for scheduler
    from_noise: bool 
        Whether to learn mapping from Gaussian noise or an initial structure
    """
    def __init__(
            self, 
            model,
            variant: str="refine",
            mask_frzn_atoms: bool=True,
            rtol: float = 1e-7, 
            atol: float = 1e-9,
            ode_method: str = "dopri5",
            track_grad_norms: bool=False,
            input_file=None,
            output_file=None,
            **hparams,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore=["model", "mask_frzn_atoms", "rtol", "atol", "ode_method", "track_grad_norms"])  # all __init__ args saved as hyperparameters
        self.model = model
        self.loss_fn = nn.MSELoss() # mean squared error loss function 
        self.variant = variant
        self.mask_frzn_atoms = mask_frzn_atoms
        self.rtol = rtol
        self.atol = atol 
        self.ode_method = ode_method
        self.track_grad_norms = track_grad_norms
        self.input_file = input_file
        self.output_file = output_file
        self.relaxed_file = None
        
        # Diagnostic studies on output structures
        self.diagnostics = True
        self.bad_iter, self.good_ter, self.iter = 0, 0, 0
        self.DS0 = []
        self.DS1 = []
        self.DS2 = []
        self.DS3 = []
        self.DS4 = []
        self.DS5 = []
        self.DS6 = []
        self.DS7 = []

    def forward(self, data):
        return self.model(data)
    
    def _transport_loss(self, input_data: Data) -> torch.Tensor: 

        data, u = self.path(input_data=input_data)
        u_hat = self(data) # N, 3

        if self.mask_frzn_atoms and self.variant == "refine":
            self.free_atoms = (1 - data.fixed).unsqueeze(1) # N
            loss, median_loss, max_loss, max_loss_idx = self.masked_loss(a=u_hat, b=u, mask=self.free_atoms) 
        else: 
            loss = self.loss_fn(u_hat, u)  
        return loss, median_loss, max_loss, max_loss_idx
    
    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss, median_loss, max_loss, _ = self._transport_loss(data)
        
        self.log( # log train loss per epoch 
            "train/loss",
            loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log( # log train loss per epoch 
            "train/median_loss",
            median_loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
    
        self.log( # log train loss per epoch 
            "train/max_loss",
            max_loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        return loss
        
    
    def validation_step(self, data: Data, batch_idx):
        loss, median_loss, max_loss, _ = self._transport_loss(data)
        
        self.log(
            "val/loss", 
            loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log(
            "median_val/loss", 
            median_loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log(
            "val/max_loss", 
            max_loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        return loss
    
    def test_step(self, data: Data, batch_idx):
        # compute performance metrics/diagnostics
        self.iter += 1
        loss, median_loss, max_loss, max_loss_idx = self._transport_loss(data) # avergae, median, argmax
    
        # define time interval for the IVP
        t = torch.linspace(0., 1., 100, 
                        device=data.pos.device, dtype=data.pos.dtype)

        if self.diagnostics:
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
                    delta_max_ads = __class__.pbc_aware_pos(
                    delta=max_atom_pos - adsorbate_pos,
                    cell=data.cell,
                    batch=torch.zeros(K, dtype=torch.long),
                    naive=True,
                    avg_xyz=True,
                    )
                    # collect diagnostics for surface atom 
                    self.DS0.append(float(delta_max_ads.min())) # distance between atom and adsorbate
                    
                 # collect diagnostics   
                self.DS1.append(int(max_loss_z)) # atomic number 
                self.DS2.append(int(max_loss_tag)) # tag (surface, bulk, adsorbate)
                
                """
                Compute statistics for max distance moved during relaxation 
                """
                # pbc aware displacement from relaxation 
                delta = __class__.pbc_aware_pos(
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
                
                 # collect diagnostics 
                self.DS3.append(int(max_distance_z)) # atomic number 
                self.DS4.append(int(max_distance_tag)) # tag (surface, bulk, adsorbate)
                self.DS5.append(float(max_distance)) # max distance moved during relaxation 
                self.DS6.append(float(distance_of_max_loss)) # max distance moved of atom with max loss 
                
                """
                Generate structures and write files for problematic losses
                
                self.output_file = f"strucs/BAD-output{self.bad_iter}_{loss}_{max_loss}_{max_loss_idx}_{max_loss_z}_{max_loss_tag}.vasp"
                self.input_file = f"strucs/BAD-input{self.bad_iter}.vasp"
                self.relaxed_file = f"strucs/BAD-relaxed{self.bad_iter}.vasp"
                opt_graph, struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse = self.sampler(data, t=t)
                """
                struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse = 0, 0, 0, 0
            
                # print statements 
                print(f"High velocity field loss {loss} ❌ || total count: {self.bad_iter}/{self.iter}")
                
                # collect diagnostics 
                self.DS3.append(int(max_distance_z)) # atomic number 
                self.DS4.append(int(max_distance_tag)) # tag (surface, bulk, adsorbate)
                self.DS5.append(float(max_distance)) # max distance moved during relaxation 
                self.DS6.append(float(distance_of_max_loss)) # max distance moved of atom with max loss 
                self.DS7.append(float(avg_distance)) # average distance moved of all atoms in struc 
                
                # save outputs to .npy 
                if self.iter == 460328 and not os.path.exists("loss-tag.npy"):
                    np.save("loss-displacement_from_ads", np.array(self.DS0))
                    np.save("loss-atomic_number", np.array(self.DS1))
                    np.save("loss-tag", np.array(self.DS2))
                    np.save("distance-atomic_number", np.array(self.DS3))
                    np.save("distance-tag", np.array(self.DS4))
                    np.save("distance-max", np.array(self.DS5))
                    np.save("distance-max_loss", np.array(self.DS6))
                    np.save("avg_distance", np.array(self.DS7))
            else: 
                struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse = 0, 0, 0, 0
        else: 
            opt_graph, struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse = self.sampler(data, t=t)
                
        self.log(
            "test/loss", 
            loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log(
            "median/test/loss", 
            median_loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log(
            "test/max_loss", 
            max_loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "test/struc_rmse", 
            struc_rmse, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log(
            "test/struc_rmdse", 
            struc_rmdse, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "test/null_struc_rmse", 
            null_struc_rmse, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        self.log(
            "test/null_struc_rmdse", 
            null_struc_rmdse, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        out_dir = self.trainer.default_root_dir
        os.makedirs(out_dir, exist_ok=True)
        
        return loss    
    
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            betas=self.hparams.betas, 
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
        )
        
        interval = "step"
        num_training_steps = int(self.trainer.estimated_stepping_batches) # estimated number of trainig steps 
        num_warmup_steps = int(self.hparams.warmup_proportion * num_training_steps)
        
        lr_scheduler = str(self.hparams.lr_scheduler)
        if lr_scheduler == "cosine_w_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.hparams.num_cycles
            )
        
        elif lr_scheduler == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=num_training_steps,
                eta_min=1e-3*self.hparams.lr,
            )
        
        elif lr_scheduler == "polynomial_w_warmup":
            if self.hparams.num_decay_steps > 0:
                anneal_total_steps = self.hparams.num_decay_steps + num_warmup_steps
            else:
                anneal_total_steps = num_training_steps
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=anneal_total_steps,
                lr_end=1e-3*self.hparams.lr,
                power=self.hparams.power
            )

        elif lr_scheduler == "polynomial_decay":
            scheduler = PolynomialLR(
                optimizer=optimizer, 
                total_iters=num_training_steps, 
                power=self.hparams.power,
            )
        else:
            raise ValueError(f"Scheduler not specified")
 
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1, 
            }
        }
        
    def on_before_optimize_step(self, optimizer): 
        if self.track_grad_norms:
                # Compute total gradient norm without clipping
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float('inf')
                    )
                # Log the gradient norm for monitoring
                self.log("unclipped grad norm", grad_norm)
        
        
