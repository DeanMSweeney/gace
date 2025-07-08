import torch 
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from adsflow.models.gemnet.gemnet_flow import GemNetFlow

class LightFlowMatch(pl.LightningModule):
    """
    Functionality 
    -------------
    This class defines the training algorithm for the conditional flow matching model using optimal transport 
    
    Arguments
    ---------
    model_cfg: dict 
        Model configuration 
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
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.from_noise = from_noise

    def forward(self, data):
        return self.model(data)
    
    def training_step(self, data, batch_idx):
        x1 = data.pos_relaxed.clone() # extract relaxed positions
        del data.pos_relaxed # remove from the data object

        if self.from_noise:
            x0 = torch.randn_like(x1) # sample Gaussian distribution
        else: 
            x0 = data.pos_init.clone() # sample initial positions
            del data.pos_init # remove from the data object 

        # optimal transport
        t_graph = torch.rand(len(data.natoms), 1, device=x1.device)
        t = t_graph[data.batch] # broadcast t (Natoms per graph * batch_size, 3)
        xt = t * x1 + (1 - t) * x0 # interpolate between x0 and x1
        data.pos = xt # rewrite data.pos as the sample at time t
        data.t = t 

        # regress velocity field and compute loss 
        u_hat = self(data)
        u = x1 - x0 
        loss = self.loss_fn(u_hat, u) # compute training loss 

        self.log(
            "train/loss_epoch", loss,
            on_step=False, 
            on_epoch=True, 
            prog_bar=True
        )
        return loss
    
    def on_after_backward(self) -> None:
        missing = [
            name for name, p in self.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        if missing:
            # print to both stdout and the W&B log
            print("⚠️ Missing grads for:", missing)
            self.logger.experiment.log({"missing_grad_params": missing})
    
    def validation_step(self, data, batch_idx):
        x1 = data.pos_relaxed.clone() # extract relaxed positions (consider removing from the data object)
        del data.pos_relaxed
        
        if self.from_noise:
            x0 = torch.randn_like(x1) # sample Gaussian distribution
        else: 
            x0 = data.pos_init.clone() # sample initial positions
            del data.pos_init # remove from the data object 
        
        # optimal transport
        t_graph = torch.rand(len(data.natoms), 1,  device=data.pos.device)
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


############################################################
                        # Run model 
############################################################                     
if __name__ == "__main__":

    from torch_geometric.loader import DataLoader
    from adsflow.utils.data_handling import OCP_DataHandler
    from pytorch_lightning.callbacks import ModelCheckpoint
    import json

    attention = False # configure for attention mechanism via EquiformerV2
    if not attention:
        with open("gemnet_cfg.json") as f:
            model_cfg = json.load(f) # load in gemnet configuration file 
        model = GemNetFlow(**model_cfg)

    # read and build data objects with torch_geometric DataLoader
    oc20_train = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"
    oc20_val = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/val_id/data.lmdb"
    train_set = OCP_DataHandler(oc20_train, pbc=True)
    val_set = OCP_DataHandler(oc20_val, pbc=True)

    # set batch size here 
    B_n = 16

    # load training set
    train_loader = DataLoader(
        train_set, 
        batch_size=B_n,
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
    )

    # load validation set
    val_loader = DataLoader(
        val_set,
        batch_size=B_n,
        num_workers=4,
        pin_memory=True,
        )

    # compile lightning model and trainer
    lightning = LightFlowMatch(model, lr=1e-4)

    # regiester wandb logger
    wandb_logger= WandbLogger(
        project="adsflow",
        name="gauntlet_02-sonic",
        config={
            "lr": 1e-4, 
            **model_cfg
        }, 
        log_model="all",
    )

    wandb_logger.watch(lightning.model, log="all", log_freq=50)

    # incorporate model checkpints 
    checkpoint_cb = ModelCheckpoint(
    dirpath="checkpoints",
    filename="gauntlet_02-{epoch:02d}-{step}",
    save_last=True,
    save_top_k=3,
    monitor="val/loss",
    mode="min",
    )

    # confugure trainer with n devices (GPUs) 
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=300,
        precision="16-mixed",
        accumulate_grad_batches=4, # every 4 batches, update weghts 
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
    )

    # activate trainer
    trainer.fit(lightning, train_loader, val_loader)
    wandb.finish()
    

