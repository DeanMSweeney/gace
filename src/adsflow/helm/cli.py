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
import os, psutil 
import torch
from torch import optim
from torch.utils.data import random_split
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau, SequentialLR
import torch.nn as nn
import torch.distributed as dist

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
from adsflow.utils.builder import build_graph_from_atoms
from adsflow.helm.callbacks import EMACallback
from adsflow.utils.lmdb_loader import LazyLmdbDataset
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

    if cfg.mode == "eval":
        cfg.precision = 32 
    
    # === Reproducibility & Performance Setup ===
    # Seed RNG for reproducibility across multi-GPU training
    pl.seed_everything(cfg.seed, workers=True)

    # Enable TF32 for faster training on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")
    pin_memory = True if cfg.accelerator == "gpu" else False

    # === Model Initialization ===
    # Instantiate the appropriate flow matching model based on regressor type
    model = hydra.utils.instantiate(cfg.model_cfg, _convert_="partial")

    # Print configuration at rank 0 only
    declare(f">>> Training and model configurations:\n {OmegaConf.to_yaml(cfg)}")

    # === Callbacks Configuration ===
    # ModelCheckpoint: Save tp-3 models based on validation loss
    callbacks = [(ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    ))]
    # LearningRateMonitor: Track learning rate changes during training
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if cfg.optim_cfg.ema_decay is not None:
        callbacks.append(EMACallback(decay=cfg.optim_cfg.ema_decay))
        declare(f">>> EMA enabled with decay={cfg.optim_cfg.ema_decay}")

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
        test_set = Subset(test_set, indices=range(10))
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    # --- Generation Mode ---
    # Build Data object from adsorbate molecule POSCAR + target binding energy
    elif cfg.mode == "generate" and cfg.sampler_cfg.in_POS:
        declare(">>> Activated Generation mode with user adsorbate input")
        mol_atoms = ase.io.read(cfg.sampler_cfg.in_POS)
        initial_graph = build_graph_from_atoms(
            mol_atoms=mol_atoms,
            binding_energy=cfg.sampler_cfg.target_binding_E,
            num_prim_atoms=cfg.sampler_cfg.num_prim_atoms,
        )
        gen_loader = DataLoader(
            dataset=[initial_graph],
            batch_size=1,
            num_workers=0,
            pin_memory=pin_memory,
        )
        
    # === Lightning Module Setup ===
    # Wrap model in LightningModule for training orchestration
    lightning = LightFlowMatch(
        model=model,
        variant=cfg.variant,
        optimal_transport=cfg.optimal_transport,
        informed_priors=cfg.informed_priors,
        operate_on_torus=cfg.operate_on_torus,
        condition_on_binding_E=cfg.condition_on_binding_E,
        standardize_ads_pos=cfg.standardize_ads_pos,
        mode=cfg.mode,
        track_grad_norms=cfg.track_grad_norms,
        x_matching=cfg.x_matching,
        schedule_type=cfg.interpolant_schedule,
        accelerator=cfg.accelerator,
        seed=cfg.seed,
        **cfg.sampler_cfg,           # All sampler/generation/MLFF settings
        **cfg.optim_cfg,     # Optimizer hyperparameters (lr, weight_decay, etc.)
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
        max_epochs=cfg.optim_cfg.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,  # Gradient accumulation steps
        val_check_interval=cfg.val_check_interval,  # Validation frequency
        log_every_n_steps=10,
        gradient_clip_val=1,  # Gradient clipping value
    )

   # === Checkpoint Loading ===
    ckpt_path_for_trainer = cfg.ckpt_path 
    # In your main training script:
    if cfg.ckpt_path and not cfg.resume_training:
        declare(f">>> Loading CLEAN preprocessed weights from: {cfg.ckpt_path}")
        # Mock zatom so its checkpoint can be loaded without installing the package.
        # Uses a meta path finder so arbitrary-depth submodule refs (e.g.
        # zatom.models.architectures.Foo) are handled without knowing them in advance.
        import sys, types

        class _ZatomModule(types.ModuleType):
            """Module that returns dummy classes for any attribute (for pickle)."""
            __path__ = []
            def __getattr__(self, name):
                # Never intercept dunder attrs — let Python use its defaults.
                # This prevents __file__ etc. from becoming dummy classes,
                # which breaks inspect/pydantic downstream.
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                cls = type(name, (), {"__module__": self.__name__})
                setattr(self, name, cls)
                return cls

        class _ZatomFinder:
            def find_module(self, fullname, path=None):
                if fullname == "zatom" or fullname.startswith("zatom."):
                    return self
            def load_module(self, fullname):
                if fullname in sys.modules:
                    return sys.modules[fullname]
                mod = _ZatomModule(fullname)
                mod.__package__ = fullname
                mod.__file__ = "<zatom-mock>"  # must be a string for inspect
                sys.modules[fullname] = mod
                return mod

        if not any(type(f).__name__ == "_ZatomFinder" for f in sys.meta_path):
            sys.meta_path.insert(0, _ZatomFinder())

        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")

        # Zatom saves its model at self.model.model (two levels), giving keys
        # like "model.model.transformer.*".  Our module uses self.model (one
        # level), expecting "model.transformer.*".  Strip the extra prefix.
        state_dict = checkpoint["state_dict"]
        remapped = {
            ("model." + k[len("model.model."):] if k.startswith("model.model.") else k): v
            for k, v in state_dict.items()
        }
        load_result = lightning.load_state_dict(remapped, strict=False)

        #  DELETE CHECKPOINT TO FREE MEMORY
        del checkpoint
        torch.cuda.empty_cache()  # Clear GPU cache if using CUDA

        # Verification Reports
        if load_result.missing_keys:
            print("\n--- MISSING KEYS (Not w/in Checkpoint ⭐) ---")
            for key in load_result.missing_keys:
                print(f"  [!] {key}")

        if load_result.unexpected_keys:
                print("\n--- UNEXPECTED KEYS (Ignored from Checkpoint ⭐) ---")
                for key in load_result.unexpected_keys:
                    print(f"  [?] {key}")
        
        declare(">>> Model weights initialized. Training/Eval will proceed.")
        ckpt_path_for_trainer = None


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
        )
    elif cfg.mode == "generate":
        # Generate predictions for single atomic structure
        trainer.predict(
            model=lightning,
            dataloaders=gen_loader,
        )
