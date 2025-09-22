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

import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, Tuple

from adsflow.helm import LightFlowMatch
from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.utils.lmdb_dataset import LazyLmdbDataset
from adsflow.utils.utils import ship_data

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
    precision:                       Literal[16, "16-mixed", 32] = 32

    # Checkpointing & logging
    logger:                          bool                    = False
    project:                         str                     = "NAME_PROJ"
    run_name:                        str                     = "NAME_RUN"
    ckpt_dir:                        str                     = "checkpoints"
    ckpt_path:                       Optional[str]           = None
    track_grad_norms:                bool                    = False
    run_id:                          str                     = None

    # Model run mode
    variant:                         Literal["refine", "forge"] = "refine"
    mode:                            Literal["debug", "train", "eval", "generate"] = "train"

    # Sampler settings (ODEINT)
    rtol:                            float           = 1e-7
    atol:                            float           = 1e-9
    ode_method:                      str             = "dopri5"

    # Misc
    seed:                            int            = 42
    mask_frzn_atoms:                 bool           = True 
    
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

@hydra.main(version_base=None)
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
    torch.set_float32_matmul_precision("high")
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
        mask_frzn_atoms=cfg.mask_frzn_atoms,
        rtol=cfg.rtol, 
        atol=cfg.atol,
        ode_method=cfg.ode_method,
        track_grad_norms=cfg.track_grad_norms,
    )

    # compile W&B logger when logger is activated and generation mode is off 
    if cfg.logger and cfg.mode != "generate": 
        logger = WandbLogger(
            project=cfg.project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            log_model=False,
            resume="allow",
            id=cfg.run_id,
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
        gradient_clip_val=1.0,
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
    
   
