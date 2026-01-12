import os
import torch
from torch import optim 
from torch.utils.data import random_split
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau, SequentialLR
import torch.nn as nn
import pytorch_lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import LearningRateMonitor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset

import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, Tuple

from adsflow.helm.lightning import LightFlowMatch
from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.equiformer_v2.equiformerv2_flow import EquiformerV2_Flow
from adsflow.utils.lmdb_loader import LazyLmdbDataset
from adsflow.utils.utils import ship_data
from adsflow.helm.config import Config

from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs

import ase 
import ase.io

# print model/trianing config at rank 0 using decorator
@rank_zero_only
def declare(statement) -> None:
    print(statement)
    return 

def main(cfg: Config) -> None:
    # instantiate Python defaults
    defaults = Config()
    defaults_cfg  = OmegaConf.create(asdict(defaults))

    # merge YAML-loaded cfg on top of defaults
    cfg = OmegaConf.merge(defaults_cfg, cfg)

    # seed for reproducibility & multi-gpu weight updates
    pl.seed_everything(cfg.seed, workers=True)
    
    # enable TF32 on matmuls and memory pinning
    torch.set_float32_matmul_precision("high")
    pin_memory = True if cfg.accelerator == "gpu" else False # configured for GPU

    # build model with provided configuration file
    if cfg.regressor == "EquiformerV2":
        model: EquiformerV2_Flow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")
    elif cfg.regressor == "GemNet-OC":
        model: GemNetFlow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")

    # print configuration file at rank 0
    declare(f">>> Training and model configurations:\n {OmegaConf.to_yaml(cfg)}")

    # configure model checkpoint callbacks for writing data to .ckpt 
    callbacks = [(ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    ))]
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # load datasets
    if cfg.num_workers is None: 
        cfg.num_workers = max(1, os.cpu_count() - 2) # leave 2 cores free 

    if cfg.mode == "debug" and cfg.oc20_mini: 
        declare(f">>> Activated Debug mode with OC20-mini: {cfg.oc20_mini} 🕷️") 

        # build trianing set 
        p = {"src": cfg.oc20_mini}
        dataset = LazyLmdbDataset(p)
        train_set, val_set = random_split(dataset, [0.80, 0.20])

        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            shuffle=True, 
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

        #callbacks = [] # deactivate call backs for debugging 

    elif cfg.mode == "train" and cfg.oc20_train:
        declare(f">>> Activated Train mode with OC20: {cfg.oc20_train} 🦛")

        # build training set
        pt = {"src": cfg.oc20_train}
        train_set = LazyLmdbDataset(pt)
        train_loader = DataLoader(
            dataset=train_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            shuffle=True, 
        )
        
        # build validation set 
        pv = {"src": cfg.oc20_val}
        val_set = LazyLmdbDataset(pv)
        val_loader = DataLoader(
            dataset=val_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    elif cfg.mode == "eval" and cfg.oc20_test:
        declare(f">>> Activated Evaluation mode with OC20-test: {cfg.oc20_test} 🦉")
        ptst = {"src": cfg.oc20_test}
        test_set = LazyLmdbDataset(ptst)
        test_set = Subset(test_set, indices=range(100))
        test_loader = DataLoader(
            dataset=test_set, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
    elif cfg.mode == "generate" and cfg.in_POS: 
        declare(">>> Activated Generation mode with single user input 🧚‍♀️")
        atomic_struc = ase.io.read(cfg.in_POS)
        poscar_to_data = AtomsToGraphs(radius=12).convert(atoms=atomic_struc)
        gen_loader = DataLoader(
            dataset=[poscar_to_data], 
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        
    # compile lightning module 
    lightning = LightFlowMatch(
        model=model, 
        variant=cfg.variant, 
        mode=cfg.mode,
        track_grad_norms=cfg.track_grad_norms,
        mask_frzn_atoms=cfg.mask_frzn_atoms,
        rtol=cfg.rtol, 
        atol=cfg.atol,
        ode_method=cfg.ode_method,
        in_POS=cfg.in_POS,
        gen_POS=cfg.gen_POS,
        rlxd_POS=cfg.rlxd_POS, 
        config_vasp=cfg.config_vasp,
        threshold=cfg.threshold, 
        **cfg.optimizer_hparams,
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
        callbacks=callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        strategy=cfg.strategy,
        precision=cfg.precision,
        max_epochs=cfg.optimizer_hparams.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=10,
        gradient_clip_val=1,
    )

    # handle checkpoint loading
    ckpt_path_for_trainer = cfg.ckpt_path
    if cfg.ckpt_path and not cfg.resume_training:
        declare(f">>> Loading model weights only from: {cfg.ckpt_path} 🔄")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        lightning.load_state_dict(checkpoint["state_dict"])
        declare(">>> Model weights loaded. Training will restart from scratch (epoch 0, new optimizer state)")
        ckpt_path_for_trainer = None  # don't pass checkpoint to trainer

    # fit the model
    if cfg.mode == "debug" or cfg.mode == "train":
        trainer.fit(
            model=lightning,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path_for_trainer, # use for model training restart and evaluation
        )
    elif cfg.mode == "eval": 
        trainer.test(
            model=lightning,
            dataloaders=test_loader,
            ckpt_path=cfg.ckpt_path, # use for model training restart and evaluation
        )
    elif cfg.mode == "generate":
        trainer.predict(
            model=lightning,
            dataloaders=gen_loader,
            ckpt_path=cfg.ckpt_path
        )
