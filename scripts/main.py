import os
import torch
from torch import optim
from torch.utils.data import random_split
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau, SequentialLR
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torchdiffeq import odeint

import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, Tuple

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
    oc20_train:                      Optional[str] = None
    oc20_val:                        Optional[str] = None
    oc20_test:                       Optional[str] = None
    oc20_mini:                       Optional[str] = None

    # DataLoader settings
    batch_size:                      int = 8
    num_workers:                     Optional[int] = None

    # Training hyperparameters/settings
    lr:                              float = 1e-3
    max_epochs:                      int   = 3000
    accumulate_grad_batches:         int   = 4
    val_check_interval:              int   = 1000

    # Compute architecture
    accelerator:                     str = "gpu"
    devices:                         int = 1
    num_nodes:                       int = 1
    strategy:                        str = "auto"
    precision:                       Literal[16, "16-mixed", "32-true"] = "16-mixed"

    # Checkpointing & logging
    logger:                          bool                    = False
    project:                         str                     = "NAME_PROJ"
    run_name:                        str                     = "NAME_RUN"
    ckpt_dir:                        str                     = "checkpoints"
    ckpt_path:                       Optional[str]           = None

    # Model run mode
    mode:                            Literal["debug", "train", "eval", "generate"] = "train"

    # Sampler settings (ODEINT)
    rtol:                            float           = 1e-7
    atol:                            float           = 1e-9
    ode_method:                      str             = "dopri5"

    # Misc
    seed:                            int            = 42
    from_noise:                      bool           = False


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
            from_noise: bool = False,
            rtol: float = 1e-7, 
            atol: float = 1e-9,
            ode_method: str = "dopri5",
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore="model")  # all __init__ args saved as hyperparameters
        self.model = model
        self.loss_fn = nn.MSELoss() # mean squared error loss function 
        self.from_noise = from_noise
        self.rtol = rtol
        self.atol = atol 
        self.ode_method = ode_method

    def forward(self, data):
        return self.model(data)
    
    def _transport_loss(self, input_data: Data) -> torch.Tensor: 
        data = input_data.clone()

        x1 = data.pos_relaxed.clone() # sample relaxed positions and discard

        if self.from_noise:
            x0 = torch.randn_like(x1) # sample noise 
        else:
            x0 = data.pos.clone() # sample initial positions and discard 

        t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device) # (batch_size, 1)
        t = t_per_graph[data.batch] # (num_atoms, 1)

        xt = x1 * t + (1 - t) * x0 # define path via linear interpolation
        data.pos = xt # reassign xt to pos 
        data.pos_init = x0
        data.t = t

        u_hat = self(data) # compute velocity field 
        u = x1 - x0
        loss = self.loss_fn(u_hat, u)

        return loss
    
    def _sampler(
            self, 
            input_data: Data,
            
    ) -> Union[Data, Tuple[Data, torch.Tensor]]:
        
        data = input_data.clone()
        def dx_dt(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            """
            Wrap Pytorch model in a function with format: f(t, y)
            """
            data.pos = xt
            data.t = t.view(1,1).expand(len(xt), -1)
            u = self(data)
            print(u[0])
            print((data.pos_relaxed - input_data.pos)[0])
            return u

        # define time interval for the IVP
        t = torch.linspace(0., 1., 10, 
                        device=data.pos.device, dtype=data.pos.dtype)
        xt = odeint(
            func=dx_dt, 
            y0=input_data.pos,
            t=t,
            rtol=self.rtol,
            atol=self.atol,
            method=self.ode_method,
        )
        data.pos_generated = xt[-1] # write generate possitional coordinates to pos_gen attribute 

        data.pos = input_data.pos

        if hasattr(data, "pos_relaxed"):
            struc_rmse = torch.sqrt(self.loss_fn(data.pos_generated, data.pos_relaxed))
            null_struc_rmse = torch.sqrt(self.loss_fn(data.pos, data.pos_relaxed))
            return data, struc_rmse, null_struc_rmse
        else: 
            return data
 
    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss = self._transport_loss(data)

        self.log( # log train loss per epoch 
            "train/loss_epoch",
            loss, 
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

        return loss
    
    def validation_step(self, data: Data, batch_idx):
        loss = self._transport_loss(data)
        opt_graph, struc_rmse = self._sampler(data)

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
        loss = self._transport_loss(data)
        opt_graph, struc_rmse, null_struc_rmse = self._sampler(data)

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
            

# print model/trianing config at rank 0 using decorator
@rank_zero_only
def declare(statement) -> None:
    print(statement)
    return 

@rank_zero_only
def model_banner(model) -> None:
    print(">>> Model Banner \n")
    model.print_banner() # print model banner
    return 

@hydra.main(
    version_base=None,        
    config_path="conf", # ./conf
    config_name="config",
)

def main(cfg: Config) -> None:
    # 1) instantiate Python defaults
    defaults = Config()

    # 2) convert dataclass to dict, then to DictConfig
    defaults_dict = asdict(defaults)
    defaults_cfg  = OmegaConf.create(defaults_dict)

    # 3) merge YAML-loaded cfg on top of defaults
    cfg = OmegaConf.merge(defaults_cfg, cfg)

    # seed for reproducibility & multi-gpu weight updates
    pl.seed_everything(cfg.seed, workers=True)
    
    # enable TF32 on matmuls and memory pinning
    torch.set_float32_matmul_precision('medium')
    pin_memory = True if cfg.accelerator == "gpu" else False # configured for GPU

    # build model with  provided configuration file
    model: GemNetFlow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")
    model_banner(model)


    # print configuration file at rank 0
    declare(f">>> Training and model configurations:\n {OmegaConf.to_yaml(cfg)}")

    # configure model checkpoint callbacks
    checkpoint_cb = [(ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    ))]

    # load datasets
    if cfg.num_workers is None: 
        cfg.num_workers = max(1, os.cpu_count() - 2) # leave 2 cores free 

    if cfg.mode == "debug" and cfg.oc20_mini: 
        declare(f">>> Activated Debug mode with OC20-mini: {cfg.oc20_mini} 🕷️")

        # build trianing set 
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

        checkpoint_cb = [] # deactivate call backs fo debugging 

    elif cfg.mode == "train" and cfg.oc20_train:
        declare(f">>> Activated Train mode with OC20: {cfg.oc20_train} 🦛")

        # build training set
        pt = {"src": cfg.oc20_train}
        train_set = LazyLmdbDataset(pt)
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        
        # build validation set 
        pv = {"src": cfg.oc20_val}
        val_set = LazyLmdbDataset(pv)
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    elif cfg.mode == "eval" and cfg.oc20_test:
        declare(f">>> Activated Evaluation mode with OC20-test: {cfg.oc20_test} 🦉")
        ptst = {"src": cfg.oc20_test}
        test_set = LazyLmdbDataset(ptst)
        test_loader = DataLoader(
            test_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
    elif cfg.mode == "generate": 
        declare(">>> Activated Generation mode with user input 🧚‍♀️")

    # compile lightning module 
    lightning = LightFlowMatch(
        model=model, 
        lr=cfg.lr,
        from_noise=cfg.from_noise,
        rtol=cfg.rtol, 
        atol=cfg.atol,
        ode_method=cfg.ode_method
    )

    # compile W&B logger when logger is activated and generation mode is off 
    if cfg.logger and cfg.mode != "generate": 
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
    if cfg.mode == "debug" or cfg.mode == "train":
        trainer.fit(
            model=lightning,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.ckpt_path, # use for model training restart and evaluation
        )
    elif cfg.mode == "eval": 
        trainer.test(
            model=lightning,
            dataloaders=test_loader,
            ckpt_path=cfg.ckpt_path, # use for model training restart and evaluation
        )
    elif cfg.mode == "generate":
        trainer.predict(model=lightning)
    
if __name__ == "__main__":
    main()
    
   
