import torch
from torch import optim
from torch.utils.data import random_split
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
from typing import Optional, Literal

from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.gemnet.gemnet_base import GemNetOC
from adsflow.utils.lmdb_dataset import LazyLmdbDataset
from fairchem.core.datasets.lmdb_dataset import LmdbDataset

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
    logger: bool
        Whether to use WandB logger => Default: False
    project: str 
        Project name (for logging)
    run_name: str
        Iteration name (for logging/checkpointing)
    ckpt_dir: str
        File path for storing model checkpoints 
    ckpt_path: str
        Filepath to restart from trained model
    mode: str
        Run mode: Debug, train, eval, or prod => Default: "train"
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
    oc20_test:  str = MISSING
    oc20_mini:  str = MISSING

    # DataLoader settings
    batch_size:      int = MISSING
    num_workers:     int = MISSING

    # Training hyperparameters/settings
    lr:                       float = 1e-3
    max_epochs:               int   = 3000
    accumulate_grad_batches:  int   = 4
    val_check_interval:       int   = 1000

    # Compute architecture
    accelerator: str = MISSING
    devices:     int = MISSING
    num_nodes:   int = MISSING
    strategy:    str = "auto"
    precision:   Literal[16, "16-mixed", "32-true"] = "16-mixed"

    # Checkpointing & logging
    logger:           bool         = False
    project:          str          = MISSING
    run_name:         str          = MISSING
    ckpt_dir:         str          = "checkpoints"
    ckpt_path:        Optional[str] = None

    # Model run mode
    mode:             Literal["debug", "train", "eval", "prod"] = "train"

    # Misc
    seed:             int            = 42
    flow_matching:    bool           = True
    from_noise:       bool           = False


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

            if self.from_noise:
                x0 = torch.randn_like(x1) # sample noise 
            else:
                x0 = data.pos.clone() # sample initial positions and discard 

            t_graph = torch.rand(len(data.natoms), 1, device=x1.device) # t~U for each graph
            t = t_graph[data.batch] # expand t to each atom of graph 

            xt = x1 * t + (1 - t) * x0 # interpolation scheme 
            data.pos = xt # reassign xt to pos 
            data.pos_init = x0
            data.t = t

            u_hat = self(data) # compute velocity field 
            u = x1 - x0
            loss = self.loss_fn(u_hat, u)

            struc_mse = self.loss_fn(x1, xt) # compute rmse between optimized and generate structures
            struc_rmse = torch.sqrt(struc_mse)
        else: 
            F = data.force
            F_hat = self(data)
            loss = self.loss_fn(F_hat, F) # compute MSE training loss
            struc_rmse = []
          
        return loss, struc_rmse
        

    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss, struc_rmse = self._compute_loss(data)

        self.log( # log train loss per epoch 
            "train/loss_epoch",
            loss, 
            on_step=False, 
            on_epoch=True,
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log( # log structural rmse per epoch 
            "train/struc_rmse", 
            struc_rmse, 
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log( # log train loss per step. 
            "train/loss_step", 
            loss, 
            on_step=True, 
            on_epoch=False,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )


        self.log( # log structural rmse per epoch 
            "train/struc_rmse", 
            struc_rmse, 
            on_step=True, 
            on_epoch=False,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )


        return loss
    
    def validation_step(self, data: Data, batch_idx):
        loss, struc_rmse = self._compute_loss(data)
        self.log(
            "val/loss", 
            loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "val/struc_rmse", 
            struc_rmse, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        return loss

    def test_step(self, data: Data, batch_idx):
        loss, struc_rmse = self._compute_loss(data)
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

        return loss    
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer, 
                    mode="min", 
                    factor=0.50,
                    patience=2,
                    min_lr=1e-6,
                ),
                "monitor": "val/loss",
                "frequency": 1, 
            }
        }
            

# print model/trianing config at rank 0 uding decorator
@rank_zero_only
def print_config(cfg: DictConfig) -> None:
    print(f'>>> Training and model configurations:\n {OmegaConf.to_yaml(cfg)}')
    return 

@rank_zero_only
def model_banner(model) -> None:
    print(">>> Model Banner \n")
    model.print_banner() # print model banner
    return 

@hydra.main(
    version_base=None,        
    config_path="conf", # ./conf
    config_name="train_config",
)

def main(cfg: Config) -> None:

    # seed for reproducibility & multi-gpu weight updates
    pl.seed_everything(cfg.seed, workers=True)
    
    # enable TF32 on matmuls
    torch.set_float32_matmul_precision('medium')
    
    # build model with configuration file
    pin_memory = True if cfg.accelerator == "gpu" else False # configured for GPU

    if cfg.flow_matching:
        model: GemNetFlow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")
        model_banner(model)

    else: 
        model: GemNetOC = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")

    # print configuration file at rank 0
    print_config(cfg)

    # build trianing set 
    checkpoint_cb = []
    if cfg.mode == "debug": 
        print(">>> Activated Debug mode with OC20-mini 🕷️")
        #dataset = OCP_DataHandler(cfg.oc20_mini, pbc=True, flow_matching=cfg.flow_matching)
        p = {"src": cfg.oc20_mini}
        dataset = LmdbDataset(p)
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
    elif cfg.mode == "train": 
        print(">>> Activated Train mode with OC20 🦛")
        #train_set = OCP_DataHandler(cfg.oc20_train, pbc=True, flow_matching=cfg.flow_matching)
        pt = {"src": cfg.oc20_train}
        train_set = LazyLmdbDataset(pt)
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        
        # build validation set 
        #val_set = OCP_DataHandler(cfg.oc20_val, pbc=True, flow_matching=cfg.flow_matching)
        pv = {"src": cfg.oc20_val}
        val_set = LazyLmdbDataset(pv)
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        # configure model checkpoint callbacks during training 
        checkpoint_cb.append(ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        ))

    elif cfg.mode == "eval":
        print(">>> Activated Evaluation mode with OC20-test 🦉")
        ptst = {"src": cfg.oc20_test}
        test_set = LazyLmdbDataset(ptst)
        test_loader = DataLoader(
            test_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
    elif cfg.mode == "prod": 
        print(">>> Activated Performance mode with user input 🧚‍♀️")

    # compile lightning model 
    if cfg.ckpt_path: # if checkpoint file is provided 
        lightning = LightFlowMatch.load_from_checkpoint(
            checkpoint_path=cfg.ckpt_path, 
            strict=False,
            model=model,
            lr=cfg.lr, 
            flow_matching=cfg.flow_matching, 
            from_noise=cfg.from_noise,
        )
    else: # if checkpoint file is not provided
        lightning = LightFlowMatch(
            model=model, 
            lr=cfg.lr,
            flow_matching=cfg.flow_matching,
            from_noise=cfg.from_noise,
        )

    # compile W&B logger when logger is activated and performance mode is off 
    if cfg.logger and cfg.mode != "prod": 
        logger = WandbLogger(
            project=cfg.project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            log_model="False",
        )
    else: 
        logger = cfg.logger

    # define trainer 
    trainer = pl.Trainer(
        logger=logger,
        callbacks=checkpoint_cb,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        strategy=cfg.strategy,
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=10,
    )

    # fit the model 
    if cfg.mode == "eval": 
        trainer.test(
            model=lightning,
            dataloaders=test_loader,
        )
    elif cfg.mode == "debug" or cfg.mode == "train":
        trainer.fit(
            model=lightning,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

if __name__ == "__main__":
    main()


