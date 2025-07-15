import torch 
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.gemnet.gemnet_base import GemNetOC

class LightForce(pl.LightningModule):
    """
    Functionality 
    -------------
    This class defines the training algorithm for the conditional flow matching model using optimal transport 
    
    Arguments
    ---------
    model_cfg: dict 
        Model configuration dictionary
    lr: int
        Learning rate 
    from_noise: bool 
        Whether to learn mapping from pure Gaussian noise or an initial structure
    """
    def __init__(
            self, 
            model_cfg, 
            lr: int = 1e-3, 
            from_noise: bool = False
        ):
        
        super().__init__()
        self.save_hyperparameters() # all variables in __init__ are set as hyperparameters 
        self.model = GemNetOC(**model_cfg)
        self.loss_fn = nn.MSELoss()
        self.from_noise = from_noise

    def forward(self, data):
        return self.model(data)
    
    def training_step(self, data, batch_idx):
        F = data.force

        # regress velocity field and compute loss 
        F_hat = self(data)
        loss = self.loss_fn(F_hat, F) # compute MSE training loss 

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
        F = data.force

        # regress velocity field and compute loss 
        F_hat = self(data)
        loss = self.loss_fn(F_hat, F) # compute MSE validation loss 
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


### Run model ###
if __name__ == "__main__":

    from torch_geometric.loader import DataLoader
    from adsflow.utils.data_handling import OCP_DataHandler

    # compile GemNet-OC model parameters 
    model_cfg = {
        "num_targets": 1,
        "num_spherical": 7,
        "num_radial": 128,
        "num_blocks": 4,
        "emb_size_atom": 256,
        "emb_size_edge": 512,
        "emb_size_trip_in": 64,
        "emb_size_trip_out": 64,
        "emb_size_quad_in": 32,
        "emb_size_quad_out": 32,
        "emb_size_aint_in": 64,
        "emb_size_aint_out": 64,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_sbf": 32,
        "num_before_skip": 2,
        "num_after_skip": 2,
        "num_concat": 1,
        "num_atom": 3,
        "num_output_afteratom": 3,
        "num_atom_emb_layers": 2,
        "num_global_out_layers": 2,
        "use_pbc": True,
        "scale_backprop_forces": False,
        "cutoff": 12.0,
        "cutoff_qint": 12.0,
        "cutoff_aeaint": 12.0,
        "cutoff_aint": 12.0,
        "max_neighbors": 30,
        "max_neighbors_qint": 8,
        "max_neighbors_aeaint": 20,
        "max_neighbors_aint": 1000,
        "enforce_max_neighbors_strictly": True,
        "rbf": {"name": "gaussian"},
        "rbf_spherical": None,
        "envelope": {"name": "polynomial", "exponent": 5},
        "cbf": {"name": "spherical_harmonics"},
        "sbf": {"name": "legendre_outer"},
        "extensive": True,
        "forces_coupled": True,
        "output_init": "HeOrthogonal",
        "activation": "silu",
        "scale_file": None,
        "quad_interaction": False,
        "atom_edge_interaction": True,
        "edge_atom_interaction": True,
        "atom_interaction": True,
        "scale_basis": True,
    }

    # read and build data objects with torch_geometric DataLoader
    train_full = True
    if train_full: 
        oc20_train = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"
        oc20_val = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/val_id/data.lmdb"
        train_set = OCP_DataHandler(oc20_train, pbc=True, flow_matching=False)
        val_set = OCP_DataHandler(oc20_val, pbc=True, flow_matching=False)

        train_loader = DataLoader(
            train_set, 
            batch_size=32,
            shuffle=True,
            num_workers=4, 
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=32,
            num_workers=4,
            pin_memory=True,
        )
    else: 
        dataset = OCP_DataHandler("/workspace/datasets/mini/data.lmdb", pbc=True)
        loader = DataLoader(dataset, batch_size=10)


    # compile lightning model and trainer
    lightning = LightForce(model_cfg, lr=1e-4)

    wandb_logger= WandbLogger(
        project="adsflow",
        name="GemNetOC-force",
        config={
            "lr": 1e-4, 
            **model_cfg
        }, 
        log_model="all",
    )

    wandb_logger.watch(lightning.model, log="all", log_freq=50)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=300,
        precision="16-mixed",
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    # train
    trainer.fit(lightning, train_loader, val_loader)
    wandb.finish()
    
