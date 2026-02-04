"""
AdsFlow CLI - Command-line interface for training and running flow matching models

This module provides the main entry point for training, evaluating, and generating
predictions using flow matching models (EquiformerV2 or GemNet-OC) on the OC20 dataset.

Modes:
    - debug: Quick iteration on 100 samples with train/val split
    - train: Full training on OC20 dataset
    - eval: Model evaluation on test set
    - generate: Generate predictions for a single POSCAR file

The script uses PyTorch Lightning for training orchestration and supports:
    - Multi-GPU/multi-node distributed training
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing and resume functionality
    - Weights & Biases logging
"""

# Core dependencies
import os
import torch
from torch import optim
from torch.utils.data import random_split
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau, SequentialLR
import torch.nn as nn

# PyTorch Lightning for training orchestration
import pytorch_lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import LearningRateMonitor

# PyTorch Geometric for graph neural networks
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset

# Hydra and OmegaConf for configuration management
import hydra
from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, Tuple

# AdsFlow modules
from adsflow.helm.lightning import LightFlowMatch
from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.equiformer_v2.equiformerv2_flow import EquiformerV2_Flow
from adsflow.utils.lmdb_loader import LazyLmdbDataset
from adsflow.utils.utils import ship_data
from adsflow.helm.config import Config

# FairChem dataset handling
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs

# ASE for atomic structure handling
import ase
import ase.io

# Decorator to print model/training config only at rank 0 (main process in distributed training)
@rank_zero_only
def declare(statement) -> None:
    print(statement)
    return 

def main(cfg: Config) -> None:
    """
    Main entry point for training, evaluation, and generation.

    Args:
        cfg: Configuration object containing model, training, and data parameters
    """
    # === Configuration Setup ===
    # Create default configuration and merge with provided config (YAML overrides defaults)
    defaults = Config()
    defaults_cfg = OmegaConf.create(asdict(defaults))
    cfg = OmegaConf.merge(defaults_cfg, cfg)

    # === Reproducibility & Performance Setup ===
    # Seed RNG for reproducibility across multi-GPU training
    pl.seed_everything(cfg.seed, workers=True)

    # Enable TF32 for faster training on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")
    pin_memory = True if cfg.accelerator == "gpu" else False

    # === Model Initialization ===
    # Instantiate the appropriate flow matching model based on regressor type
    if cfg.regressor == "EquiformerV2":
        model: EquiformerV2_Flow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")
    elif cfg.regressor == "GemNet-OC":
        model: GemNetFlow = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")

    # Print configuration at rank 0 only
    declare(f">>> Training and model configurations:\n {OmegaConf.to_yaml(cfg)}")

    # === Callbacks Configuration ===
    # ModelCheckpoint: Save top-3 models based on validation loss
    callbacks = [(ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    ))]
    # LearningRateMonitor: Track learning rate changes during training
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # === Dataset Loading ===
    # Set number of dataloader workers (leave 2 CPU cores free for system)
    if cfg.num_workers is None:
        cfg.num_workers = max(1, os.cpu_count() - 2)

    # --- Debug Mode ---
    # Use X samples for quick iteration during development
    if cfg.mode == "debug" and cfg.oc20_train and cfg.oc20_val:
        declare(f">>> Activated Debug mode with portions of OC20 train/val: {cfg.oc20_train} 🕷️")
    
        # Load small subset and split into train/val
        pt = {"src": cfg.oc20_train}
        train_set = LazyLmdbDataset(pt)

        # Load permutations 
        g = torch.Generator().manual_seed(cfg.seed)
        indices = torch.randperm(len(train_set), generator=g)[:cfg.num_debug_samples].tolist()
        train_set = Subset(train_set, indices=indices)

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        pv = {"src": cfg.oc20_val}
        val_set = LazyLmdbDataset(pv)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

        #callbacks = [] # deactivate callbacks for debugging if needed 

    # --- Train Mode ---
    # Load full OC20 training and validation datasets
    elif cfg.mode == "train" and cfg.oc20_train:
        declare(f">>> Activated Train mode with OC20: {cfg.oc20_train} 🦛")

        # Build training dataloader from LMDB database
        pt = {"src": cfg.oc20_train}
        train_set = LazyLmdbDataset(pt)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        # Build validation dataloader
        pv = {"src": cfg.oc20_val}
        val_set = LazyLmdbDataset(pv)
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    # --- Evaluation Mode ---
    # Load test dataset for model evaluation
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

    # --- Generation Mode ---
    # Load single atomic structure from POSCAR file for generation/prediction
    elif cfg.mode == "generate" and cfg.in_POS:
        declare(">>> Activated Generation mode with single user input 🧚‍♀️")
        # Read atomic structure and convert to graph representation
        atomic_struc = ase.io.read(cfg.in_POS)
        poscar_to_data = AtomsToGraphs(radius=12).convert(atoms=atomic_struc)
        gen_loader = DataLoader(
            dataset=[poscar_to_data],
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        
    # === Lightning Module Setup ===
    # Wrap model in LightningModule for training orchestration
    lightning = LightFlowMatch(
        model=model,
        variant=cfg.variant,  # Flow matching variant (e.g., conditional, optimal transport)
        mode=cfg.mode,
        track_grad_norms=cfg.track_grad_norms,
        rtol=cfg.rtol,  # Relative tolerance for ODE solver
        atol=cfg.atol,  # Absolute tolerance for ODE solver
        ode_method=cfg.ode_method,  # ODE solver method (e.g., dopri5, euler)
        in_POS=cfg.in_POS,  # Input POSCAR path for generation
        gen_POS=cfg.gen_POS,  # Generated output POSCAR path
        rlxd_POS=cfg.rlxd_POS,  # Relaxed structure output path
        config_vasp=cfg.config_vasp,  # VASP configuration for relaxation
        threshold=cfg.threshold,  # Force convergence threshold
        x_matching=cfg.x_matching,  # Position matching strategy
        schedule_type=cfg.interpolant_schedule,  # Interpolant schedule type, 
        write_eval_metrics=cfg.write_eval_metrics,  # Write evaluation metrics
        **cfg.optimizer_hparams,  # Optimizer hyperparameters (lr, weight_decay, etc.)
    )

    # === Logger Setup ===
    # Initialize Weights & Biases logger for experiment tracking (skip for generation mode)
    if cfg.logger and cfg.mode != "generate":
        logger = WandbLogger(
            project=cfg.project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            log_model=False,
            resume="allow",  # Allow resuming runs with same run_id
            id=cfg.run_id,
        )
    else:
        logger = cfg.logger

    # === Trainer Configuration ===
    # Configure PyTorch Lightning trainer with distributed training support
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator=cfg.accelerator,  # "gpu", "cpu", etc.
        devices=cfg.devices,  # Number of devices per node
        num_nodes=cfg.num_nodes,  # Number of compute nodes
        strategy=cfg.strategy,  # Distributed strategy (ddp, deepspeed, etc.)
        precision=cfg.precision,  # Mixed precision (16-mixed, 32, bf16-mixed)
        max_epochs=cfg.optimizer_hparams.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,  # Gradient accumulation steps
        val_check_interval=cfg.val_check_interval,  # Validation frequency
        log_every_n_steps=10,
        gradient_clip_val=1,  # Gradient clipping value
    )

    # === Checkpoint Loading ===
    # Handle two checkpoint loading scenarios:
    # 1. resume_training=True: Resume full training state (epoch, optimizer, lr scheduler)
    # 2. resume_training=False: Load only model weights, reset training state
    ckpt_path_for_trainer = cfg.ckpt_path
    if cfg.ckpt_path and not cfg.resume_training:
        declare(f">>> Loading model weights only from: {cfg.ckpt_path} 🔄")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        lightning.load_state_dict(checkpoint["state_dict"])
        declare(">>> Model weights loaded. Training will restart from scratch (epoch 0, new optimizer state)")
        ckpt_path_for_trainer = None  # Don't pass checkpoint to trainer (fresh start)

    # === Training/Evaluation/Generation Execution ===
    if cfg.mode == "debug" or cfg.mode == "train":
        # Train the model with periodic validation
        trainer.fit(
            model=lightning,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path_for_trainer,  # Resume from checkpoint if provided
        )
    elif cfg.mode == "eval":
        # Evaluate model performance on test set
        trainer.test(
            model=lightning,
            dataloaders=test_loader,
            ckpt_path=cfg.ckpt_path,
        )
    elif cfg.mode == "generate":
        # Generate predictions for single atomic structure
        trainer.predict(
            model=lightning,
            dataloaders=gen_loader,
            ckpt_path=cfg.ckpt_path
        )
