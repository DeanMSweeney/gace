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

from adsflow.utils.optimal_transport import OptTransport
from adsflow.utils.post_processing import analyze_loss

from typing import Optional, Literal, Union, Tuple


class LightFlowMatch(LightningModule, OptTransport):
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
            mode: str="debug",
            mask_frzn_atoms: bool=True,
            rtol: float = 1e-7, 
            atol: float = 1e-9,
            ode_method: str = "dopri5",
            track_grad_norms: bool=False,
            in_POS=None,
            gen_POS=None,
            rlxd_POS=None,
            config_vasp: bool=False,
            **hparams,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore=["model", "variant", "mode", "mask_frzn_atoms", "rtol", "atol", "ode_method", "track_grad_norms", "in_POS", "gen_POS", "rlxd_POS", "run_vasp"])  # all __init__ args saved as hyperparameters
        self.model = model
        self.variant = variant 
        self.mode = mode
        self.loss_fn = nn.MSELoss() # mean squared error loss function 
        self.mask_frzn_atoms = mask_frzn_atoms
        self.rtol = rtol
        self.atol = atol 
        self.ode_method = ode_method
        self.track_grad_norms = track_grad_norms
        self.in_POS = in_POS
        self.gen_POS = gen_POS
        self.rlxd_POS = rlxd_POS
        self.config_vasp = config_vasp
        
        # =============================================================================
        # Diagnostic studies on output model performance during evaluation
        #                   ** For model development only ** 
        self.diagnostics = False
        self.bad_iter, self.good_ter, self.iter = 0, 0, 0
       # ==============================================================================
       
    def forward(self, data):
        return self.model(data)
    
    def _transport_loss(self, input_data: Data) -> torch.Tensor: 

        data, u = self.path(input_data=input_data)
        u_hat = self(data) # N, 3

        if self.mask_frzn_atoms and self.variant == "refine":
            self.free_atoms = (1 - data.fixed).unsqueeze(1) # N
            loss, max_loss, max_loss_idx = self.masked_loss(
                input_data=data, 
                a=u_hat, 
                b=u,
                mask=self.free_atoms,
                alpha=self.hparams.alpha, 
                ramp_L=self.hparams.ramp_L,
                ramp_n_epochs=self.hparams.ramp_n_epochs,
            ) 
        else: 
            loss = self.loss_fn(u_hat, u)  
        return loss, max_loss, max_loss_idx
    
    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss, max_loss, _ = self._transport_loss(data)
        
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
        loss, max_loss, _ = self._transport_loss(data)
        
        self.log(
            "val/loss", 
            loss, 
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
        loss, max_loss, max_loss_idx = self._transport_loss(data) # avergae, median, argmax

        if self.diagnostics:
            if loss > 0.001: 
                self.bad_iter += 1 
                (self.DS0, 
                self.DS1, 
                self.DS2, 
                self.DS3, 
                self.DS4, 
                self.DS5, 
                self.DS6, 
                self.DS7) = analyze_loss(
                    data=data,
                    loss=loss, 
                    max_loss=max_loss, 
                    max_loss_idx=max_loss_idx,
                    bad_iter=self.bad_iter, 
                )
            struc_rmse, struc_rmdse, null_struc_rmse, null_struc_rmdse = 0, 0, 0, 0
        else: 
            # define time interval for the IVP
            t = torch.linspace(0., 1., 100, device=data.pos.device, dtype=data.pos.dtype)
            opt_graph, struc_rmse, null_struc_rmse = self.sample(input_data=data, t=t)
            print(f"Mean loss of batch: {loss} || Structural RMSE of batch: {struc_rmse}")
                
        self.log(
            "test/loss", 
            loss, 
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

    def on_test_end(self):
        if self.diagnostics:
            print(">>> You are running with diagnostic mode turned on ‼️")
            
            # Convert Python lists → tensors so Lightning can all_gather them
            DS0 = torch.tensor(self.DS0, dtype=torch.float32)
            DS1 = torch.tensor(self.DS1, dtype=torch.long)
            DS2 = torch.tensor(self.DS2, dtype=torch.long)
            DS3 = torch.tensor(self.DS3, dtype=torch.long)
            DS4 = torch.tensor(self.DS4, dtype=torch.long)
            DS5 = torch.tensor(self.DS5, dtype=torch.float32)
            DS6 = torch.tensor(self.DS6, dtype=torch.float32)
            DS7 = torch.tensor(self.DS7, dtype=torch.float32)
            
            # Convert gathered tensors back to 1D NumPy arrays
            def unwrap(x):
                return x.reshape(-1).cpu().numpy()

            np.save("loss-displacement_from_ads", unwrap(DS0))
            np.save("loss-atomic_number",        unwrap(DS1))
            np.save("loss-tag",                  unwrap(DS2))
            np.save("distance-atomic_number",    unwrap(DS3))
            np.save("distance-tag",              unwrap(DS4))
            np.save("distance-max",              unwrap(DS5))
            np.save("distance-max_loss",         unwrap(DS6))
            np.save("avg_distance",              unwrap(DS7))

        
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
        
        
