import os
import torch
from torch import optim 
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau, SequentialLR
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Data

from adsflow.utils.flow_matching import OptTransportMixin
from adsflow.utils.utils import ship_data


class LightFlowMatch(pl.LightningModule, OptTransportMixin):
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
            variant: str = "refine",
            mask_frzn_atoms: bool = True,
            lr: int = 1e-3, 
            weight_decay: int = 0.001,
            rtol: float = 1e-7, 
            atol: float = 1e-9,
            ode_method: str = "dopri5",
            track_grad_norms: bool = False,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore="model")  # all __init__ args saved as hyperparameters
        self.model = model
        self.loss_fn = nn.MSELoss() # mean squared error loss function 
        self.variant = variant
        self.mask_frzn_atoms = mask_frzn_atoms
        self.rtol = rtol
        self.atol = atol 
        self.ode_method = ode_method
        self.track_grad_norms = track_grad_norms
        self.naive = True



    def forward(self, data):
        return self.model(data)
    
    def _transport_loss(self, input_data: Data) -> torch.Tensor: 

        data, u = self.path(input_data=input_data)
        u_hat = self(data) # N, 3

        if self.mask_frzn_atoms and self.variant == "refine":
            self.free_atoms = (1 - data.fixed).unsqueeze(1) # N
            loss, max_loss = self.masked_loss(a=u_hat, b=u, mask=self.free_atoms) 
        else: 
            loss = self.loss_fn(u_hat, u)  
        return loss, max_loss
    
    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss, max_loss = self._transport_loss(data)
        
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
        loss, max_loss = self._transport_loss(data)
        
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
        loss, max_loss = self._transport_loss(data)
        
        # define time interval for the IVP
        t = torch.linspace(0., 1., 100, 
                        device=data.pos.device, dtype=data.pos.dtype)
        opt_graph, struc_rmse, null_struc_rmse = self.sampler(data, t=t)

        self.log(
            "test/loss", 
            loss, 
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
            "test/null_struc_rmse", 
            null_struc_rmse, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        return loss    
    
    def predict_step(self, data: Data): 
        opt_graph = self._sampler(data)
        return opt_graph
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer, 
                    mode="min", 
                    factor=0.50,
                    patience=2,
                    min_lr=1e-8,
                ),
                "monitor": "val/loss",
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
        
        
