import torch
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

class LightFlowMatch(pl.LightningModule):
    """
    Functionality 
    -------------
    This class defines the training algorithm for the conditional flow matching model using optimal transport 
    
    Arguments
    ---------
    model: class 
        Neural network based model for learning velocity field
    lr: int
        Learning rate 
    from_noise: bool 
        Whether to learn mapping from pure Gaussian noise or an initial structure
    """
    def __init__(
            self, 
            model,
            lr: int = 1e-3, 
            from_noise: bool = False
        ):
        
        super().__init__()
        self.save_hyperparameters() # all variables in __init__ are set as hyperparameters 
        self.model = model# define neutrla network
        self.loss_fn = nn.MSELoss()
        self.from_noise = from_noise

    def forward(self, data):
        return self.model(data)
    
    def training_step(self, data, batch_idx):
        x1 = data.pos_relaxed # extract relaxed positions 

        if self.from_noise:
            x0 = torch.randn_like(x1) # sample Gaussian distribution
        else: 
            x0 = data.pos_init # sample initial positions
        
        # optimal transport
        t_graph = torch.rand(len(data.natoms), 1)
        t = t_graph[data.batch] # broadcast t (Natoms per graph * batch_size, 3)
        xt = t * x1 + (1 - t) * x0 # interpolate between x0 and x1
        data.pos = xt # reqrite data.pos as the sample at time t
        data.t = t 

        # regress velocity field and compute loss 
        u_hat = self(data)
        u = x1 - x0 
        loss = self.loss_fn(u_hat, u) # compute training loss 

        self.log(
            "train/loss", loss,
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
        return loss
        
    def validation_step(self, data, batch_idx):
        x1 = data.pos_relaxed # extract unoptimized positions 

        if self.from_noise:
            x0 = torch.randn_like(x1) # sample Gaussian distribution
        else: 
            x0 = data.pos_init # sample pre-relaxed positions 
        
        # optimal transport
        t_graph = torch.rand(len(data.natoms), 1)
        t = t_graph[data.batch] # broadcast t (Natoms per graph * batch_size, 3)
        xt = t * x1 + (1 - t) * x0 # interpolate between x0 and x1
        data.pos = xt 
        data.t = t 

        # regress velocity field and compute loss 
        u_hat = self(data)
        u = x1 - x0
        loss = self.loss_fn(u_hat, u) # compute  validation loss 
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


### Testing ###

if __name__ == "__main__":

    from torch_geometric.loader import DataLoader
    from adsflow.utils.data_handling import OCP_DataHandler
    from adsflow.models.gemnet.gemnet_flow import GemNetFlow
    from adsflow.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_Flow
    import json
    import os

    attention = False # configure for attention mechanism via EquiformerV2
    if not attention:
        with open("gemnet_cfg.json") as f:
            model_cfg = json.load(f) # load in gemnet configuration file 
        model = GemNetFlow(**model_cfg)

    else:
         with open("equiformerv2_cfg.json") as f:
            model_cfg = json.load(f) # load in the equiformer configuration file 
         model = EquiformerV2_Flow(**model_cfg)

    # read and build data objects with torch_geometric DataLoader
    dataset = OCP_DataHandler("/workspace/datasets/mini/data.lmdb", pbc=True)
    loader = DataLoader(
            dataset,
            batch_size=10, 
            num_workers=15,
            pin_memory=True,
            )

    # compile lightning model and trainer
    lightning = LightFlowMatch(model, lr=1e-4)

    wandb_logger= WandbLogger(
        project="adsflow",
        name="mini-batch_01",
        config={
            "lr": 1e-4, 
            **model_cfg
        }, 
        log_model="all",
    )

    wandb_logger.watch(lightning.model, log="all", log_freq=50)
    trainer = pl.Trainer(
        max_epochs=300, 
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    # train
    trainer.fit(lightning, loader)
    wandb.finish()



