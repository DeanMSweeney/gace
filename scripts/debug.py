from dataclasses import dataclass

import torch
from torch import optim
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau, SequentialLR
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass
from typing import Optional

from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.gemnet.gemnet_base import GemNetOC
from adsflow.utils.data_handling import OCP_DataHandler



@dataclass
class Config:
    """
    This class integrates Hydra to build a dynamic hierarchical configruation 
    for the model 

    Args
    _________

    model_cfg: DictConfig 
        Configuration file (.yaml) for the run,
        including model selection and hyperparameters 
    oc20_train: str 
        File path to the training set 
    oc20_val: str
        File path to the validation set 
    oc20_mini: str
        Path to the oc20 100 dataset 
    batch_size: int
        Batch size for training 
    num_workers: int
        Cpu cores used for the DataLoader 
    lr: float
        Learning rate
    max_epochs: int 
        Max numner of trianing iterations
    accumulate_grad_batches: int 
        Number of gradients to accumulate before updating the weights 
    accelerator: str
        Device (cpu, gpu)
    num_nodes: int
        Number of nodes 
    strategy: str 
        Parallelization strategy => Default: "auto" 
    precision: str 
        Digit precision 
    debug_mode: bool
        Set to True if debugging for trianing on OC20-mini dataset => Default: False
    logger: bool
        Whether to use WandB logger => Default: False
    project: str 
        Project name (for logging)
    run_name: str
        Iteration name (for logging/checkpointing)
    ckpt_dir: str
        File path for storing model checkpoints 
    ckpt_path: str
        Filepath to previous checkpoints to restart model trianing
    seed: int
        Seed for random number generators for reproducibility => Default: 42
    flow_matching: bool 
        Whether to perform force regression or velcoity field predictions => Default: True
    from_noise: bool
        Whether to sample x0 from noise or use initial geometries => Default: False
    """
    # Hydra composition entry for model hyperparameters
    model_cfg: DictConfig = MISSING

    # Dataset paths
    oc20_train: str = MISSING
    oc20_val:   str = MISSING
    oc20_mini:  str = MISSING

    # DataLoader settings
    batch_size:      int = MISSING
    num_workers:     int = MISSING

    # Training hyperparameters
    lr:                       float = 1e-3
    max_epochs:               int   = 3000
    accumulate_grad_batches:  int   = 4

    # Compute architecture
    accelerator: str = MISSING
    devices:     int = MISSING
    num_nodes:   int = MISSING
    strategy:    str = "auto"
    precision:   str = "16-mixed"

    # Checkpointing & logging
    debug_mode:       bool         = False
    logger:           bool         = False
    project:          str          = MISSING
    run_name:         str          = MISSING
    ckpt_dir:         str          = "checkpoints"
    ckpt_path:        Optional[str] = None

    # Misc
    seed:       int  = 42
    flow_matching: bool = True
    from_noise: bool = False


class LightFlowMatch(pl.LightningModule):
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
            lr: int = 1e-3, 
            flow_matching: bool = True,
            from_noise: bool = False,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore="model")  # all __init__ args saved as hyperparameters
        self.model = model
        self.loss_fn = nn.MSELoss() # mean squared error loss function 
        self.flow_matching = flow_matching
        self.from_noise = from_noise

    def forward(self, data):
        return self.model(data)
    
    def _compute_loss(self, data: Data) -> torch.Tensor: 
        if self.flow_matching: 
            x1 = data.pos_relaxed.clone() # sample relaxed positions and discard
            del data.pos_relaxed

            if self.from_noise:
                x0 = torch.randn_like(x1) # sample noise 
            else:
                x0 = data.pos_init.clone() # sample initial positions and discard 
                del data.pos_init

            t_graph = torch.rand(len(data.natoms), 1, device=x1.device) # t~U for each graph
            t = t_graph[data.batch] # expand t to each atom of graph 

            xt = x1 * t + (1 - t) * x0 # interpolation scheme 
            data.pos = xt # reassign xt to pos 
            data.t = t

            u_hat = self(data) # compute velocity field 
            u = x1 - x0
            loss = self.loss_fn(u_hat, u)
        else: 
            F = data.force
            F_hat = self(data)
            loss = self.loss_fn(F_hat, F) # compute MSE training loss

        return loss

    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss = self._compute_loss(data)
        self.log(
            "train/loss_epoch",
            loss, 
            on_step=False, 
            on_epoch=True,
            prog_bar=True, 
            batch_size=len(data.natoms),
        )

        self.log(
            "train/loss_step", 
            loss, 
            on_step=True, 
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        return loss
    
    def validation_step(self, data: Data, batch_idx):
        loss = self._compute_loss(data)
        self.log(
            "val/loss", 
            loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # warmup phase
        warmup_epochs = 5
        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=0.05, 
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # expotential decay phase 
        decay_scheduler = ExponentialLR(optimizer=optimizer,  gamma=0.9,)

        # combined earmup + expotential decay
        combined_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, decay_scheduler], 
            milestones=[warmup_epochs],
        )

        plateau_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6, 
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": 
                {
                    "scheduler": combined_scheduler,
                    "interval": "epoch",   # warmup & switch counted in epochs
                    "frequency": 1,
                },
                "optimizer": optimizer, 
                
                "lr_scheduler":
                {
                    "scheduler": plateau_scheduler,
                    "interval": "epoch",   # also per-epoch
                    "monitor": "val/loss",
                    "frequency": 1,
                },
        }

# print model/trianing config at rank 0 uding decorator
@rank_zero_only
def print_config(cfg: DictConfig) -> None:
    print(f'>>> Training and model configurations:\n {OmegaConf.to_yaml(cfg)}')
    return 

@hydra.main(
    version_base=None,        
    config_path="conf", # ./conf
    config_name="debug_config",
)
def main(cfg: Config) -> None:

    # seed for reproducibility & multi-gpu weight updates
    pl.seed_everything(cfg.seed, workers=True)
    
    # build model with configuration file
    pin_memory = True if cfg.accelerator == "gpu" else False # configured for GPU

    if cfg.flow_matching:
        model: GemNetFlow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")
    else: 
        model: GemNetOC = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")

    # print configuration file at rank 0
    config = print_config(cfg)

    # build trianing set 
    if not cfg.debug_mode:
        train_set = OCP_DataHandler(cfg.oc20_train, pbc=True, flow_matching=cfg.flow_matching)
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        
        # build validation set 
        val_set = OCP_DataHandler(cfg.oc20_val, pbc=True, flow_matching=cfg.flow_matching)
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
    
    else: 
        print(">>> DEBUG MODE with OC20-mini")
        dataset = OCP_DataHandler(cfg.oc20_mini, pbc=True, flow_matching=cfg.flow_matching)
        train_set, val_set = random_split(dataset, [0.80, 0.20])

        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    # compile lightning model 
    lightning = LightFlowMatch(
        model=model, 
        lr=cfg.lr,
        flow_matching=cfg.flow_matching,
        from_noise=cfg.from_noise,
    )

    # compile W&B logger
    if cfg.logger:
        logger = WandbLogger(
            project=cfg.project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            log_model="all",
        )
    else: 
        logger = cfg.logger

    # configure model checkpoints 
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # define trainer 
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        strategy=cfg.strategy,
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=10,
    )

    # fit the model 
    trainer.fit(
        lightning,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path, # load model from checkpoint
    )

if __name__ == "__main__":
    main()

