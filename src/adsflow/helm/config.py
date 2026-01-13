"""
Configuration management for AdsFlow training pipeline.

This module defines configuration dataclasses using Hydra/OmegaConf for:
1. Optimizer hyperparameters (OptimizerConfig)
2. Full training pipeline settings (Config)

Configurations can be loaded from YAML files and overridden via command line.
"""

from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Union, Tuple


@dataclass
class OptimizerConfig:
    """
    Optimizer and learning rate scheduler configuration.

    Contains all hyperparameters for AdamW optimizer and LR scheduling strategies.
    """

    # ==================== Optimizer Hyperparameters ====================

    lr: float = 1e-3
    """Base learning rate for AdamW optimizer (default: 1e-3)"""

    weight_decay: float = 0.0001
    """L2 regularization coefficient for AdamW (default: 1e-4)"""

    betas: Tuple[float, float] = (0.9, 0.99)
    """Adam momentum parameters (beta1, beta2) for first and second moments
    - beta1: exponential decay rate for first moment (gradient)
    - beta2: exponential decay rate for second moment (squared gradient)
    """

    eps: float = 1e-8
    """Numerical stability epsilon for Adam denominator"""

    # ==================== Learning Rate Scheduler ====================

    lr_scheduler: Literal["polynomial_decay", "cosine_w_warmup", "cosine_annealing", "polynomial_w_warmup", "polynomial_decay"] = "cosine_w_warmup"
    """
    Learning rate scheduling strategy:
    - cosine_w_warmup: Cosine annealing with linear warmup (recommended)
    - cosine_annealing: Standard cosine annealing without warmup
    - polynomial_w_warmup: Polynomial decay with linear warmup
    - polynomial_decay: Polynomial decay without warmup
    """

    num_cycles: float = 0.50
    """Number of cosine cycles for cosine_w_warmup scheduler
    - 0.5: single descent (most common)
    - 1.0: full cosine cycle (descend then ascend)
    """

    warmup_proportion: float = 0.01
    """Fraction of training steps for linear warmup (default: 1% of total steps)
    E.g., 0.01 = first 1% of training uses linear warmup
    """

    eta_min: float = 0.0
    """Minimum learning rate for cosine annealing (lr floor)"""

    power: float = 1.0
    """Polynomial exponent for polynomial decay schedulers
    - 1.0: linear decay
    - 2.0: quadratic decay
    """

    max_epochs: int = 100
    """Maximum number of training epochs"""

    # ==================== Loss Weighting ====================

    tao: float = 0
    """Adsorbate weighting factor for loss computation
    - 0: equal weighting for all atoms
    - >0: upweight adsorbate atoms by factor tao
    E.g., tao=2 means adsorbate atoms contribute 2x more to loss
    """

    
@dataclass
class Config:
    """
    Main configuration for AdsFlow training pipeline.

    Integrates Hydra/OmegaConf for dynamic hierarchical configuration management.
    All parameters can be overridden via YAML config files or command line arguments.

    Key sections:
    - Model architecture and hyperparameters
    - Dataset paths and data loading
    - Compute resources (GPUs, distributed training)
    - Logging and checkpointing
    - Flow matching algorithm settings
    - ODE solver configuration for inference
    """

    # ==================== Model Configuration ====================

    model_cfg: DictConfig = MISSING
    """
    Model architecture configuration (loaded from YAML).
    Contains model-specific hyperparameters for:
    - EquiformerV2: hidden_channels, num_layers, attention heads, etc.
    - GemNet-OC: embedding dimensions, interaction blocks, etc.
    """

    optimizer_hparams: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer and learning rate scheduler configuration"""

    # ==================== Dataset Paths ====================

    oc20_train: Optional[str] = None
    """Path to training dataset (e.g., OC20 LMDB or pickle file)"""

    oc20_val: Optional[str] = None
    """Path to validation dataset for monitoring generalization"""

    oc20_test: Optional[str] = None
    """Path to test dataset for final evaluation"""

    # ==================== DataLoader Settings ====================

    batch_size: int = 1
    """Number of structures per batch
    - Larger batches: more stable gradients, higher GPU memory usage
    - Smaller batches: more noise, can help generalization
    Note: For graph neural networks, batch_size refers to number of graphs
    """

    num_workers: Optional[int] = None
    """Number of CPU workers for data loading (parallel preprocessing)
    - None: auto-detect based on CPU cores
    - Higher values: faster data loading but more memory usage
    """

    accumulate_grad_batches: int = 1
    """Number of batches to accumulate gradients before optimizer step
    Effective batch size = batch_size × accumulate_grad_batches
    Useful for training with large effective batch sizes on limited GPU memory
    """

    val_check_interval: int = 1.0
    """How often to run validation during training
    - 1.0: validate after every epoch
    - 0.5: validate twice per epoch
    - 2: validate every 2 epochs
    """

    # ==================== Compute Architecture ====================

    accelerator: str = "gpu"
    """Hardware accelerator: "gpu", "cpu", "tpu", "mps" (Apple Silicon)"""

    devices: int = 1
    """Number of devices (GPUs) to use per node
    - 1: single GPU training
    - >1: data parallel training across multiple GPUs
    """

    num_nodes: int = 1
    """Number of compute nodes for distributed training
    - 1: single node training
    - >1: multi-node distributed training (requires cluster setup)
    """

    strategy: str = "auto"
    """PyTorch Lightning training strategy
    - "auto": automatically select strategy based on devices/nodes
    - "ddp": DistributedDataParallel (recommended for multi-GPU)
    - "fsdp": Fully Sharded Data Parallel (for very large models)
    """

    precision: Literal[16, "16-mixed", "bf16-mixed", 32] = "bf16-mixed"
    """Training precision mode
    - "bf16-mixed": BFloat16 mixed precision (recommended, stable and fast)
    - "16-mixed": Float16 mixed precision (faster but less stable)
    - 32: Full Float32 precision (most stable but slowest)
    """

    # ==================== Logging ====================

    logger: bool = True
    """Whether to enable Weights & Biases (WandB) logging
    Logs metrics, hyperparameters, and optionally model checkpoints
    """

    project: str = "NAME_PROJ"
    """WandB project name (groups related experiments)"""

    run_name: str = "NAME_RUN"
    """Unique name for this training run (used for logging and checkpointing)"""

    run_id: str = None
    """WandB run ID for resuming existing runs (auto-generated if None)"""

    track_grad_norms: bool = True
    """Whether to log gradient norms for monitoring training stability
    Useful for detecting exploding/vanishing gradients
    """

    # ==================== Model Checkpointing ====================

    ckpt_dir: str = "checkpoints"
    """Directory for saving model checkpoints during training"""

    ckpt_path: Optional[str] = None
    """Path to checkpoint file for resuming training or evaluation
    - None: train from scratch
    - "path/to/checkpoint.ckpt": resume from checkpoint
    """

    resume_training: bool = True
    """Whether to resume training state (optimizer, scheduler, epoch) from checkpoint
    - True: resume training (continues from saved epoch)
    - False: load weights only (restarts training from epoch 0)
    """

    # ==================== Model Run Mode ====================

    variant: Literal["refine", "forge"] = "refine"
    """
    Flow matching variant:
    - "refine": learn x0 → x1 (initial to relaxed structure)
    - "forge": learn noise/random → x1 (generative modeling from scratch)
    """

    mode: Literal["debug", "train", "eval", "generate"] = "generate"
    """
    Execution mode:
    - "train": train model on dataset
    - "eval": evaluate model on test set (compute metrics)
    - "generate": generate structures from checkpoint (ODE integration)
    - "debug": debug mode with additional checks
    """

    regressor: Literal["GemNet-OC", "EquiformerV2"] = "EquiformerV2"
    """
    Neural network architecture for velocity field prediction:
    - "EquiformerV2": Equivariant transformer (recommended, state-of-the-art)
    - "GemNet-OC": Geometric message passing network (baseline)
    """

    x_matching: Literal["flow", "endpoint"] = "flow"
    """
    Loss computation mode:
    - "flow": velocity matching (predict dx/dt, standard flow matching)
    - "endpoint": position matching (predict x1 directly)
    """

    interpolant_schedule: Literal["linear", "cosine"] = "linear"
    """
    Interpolation schedule for x(t) = x0 + alpha(t) * (x1 - x0):
    - "linear": alpha(t) = t (uniform sampling in time)
    - "cosine": alpha(t) = 1 - cos^2(t*pi/2) (more samples near endpoints)
    """

    # ==================== ODE Solver Settings (for Inference) ====================

    rtol: float = 1e-7
    """Relative tolerance for adaptive ODE solver (controls accuracy)
    Smaller values → higher accuracy but slower integration
    """

    atol: float = 1e-9
    """Absolute tolerance for adaptive ODE solver (controls accuracy)
    Smaller values → higher accuracy but slower integration
    """

    ode_method: str = "dopri5"
    """
    ODE integration method for sampling (torchdiffeq):
    - "dopri5": Dormand-Prince 5th order (adaptive, recommended)
    - "euler": Forward Euler (fast but inaccurate)
    - "rk4": 4th order Runge-Kutta (fixed step, accurate)
    - "adaptive_heun": 2nd order adaptive (good speed/accuracy tradeoff)
    """

    threshold: Optional[float] = 0.20
    """Distance threshold (Angstroms) for computing evaluation metrics
    Used for Distance within Threshold (DwT) and ADwT metrics
    E.g., 0.20 = count atoms within 0.2 Angstroms of target position
    """

    # ==================== Structure Generation/Output ====================

    in_POS: Optional[str] = None
    """Output path for saving initial (unrelaxed) structures during testing
    Typically POSCAR format for visualization or DFT verification
    """

    gen_POS: Optional[str] = None
    """Output path for saving generated (model-predicted) structures during testing"""

    rlxd_POS: Optional[str] = None
    """Output path for saving relaxed (ground truth) structures during testing"""

    config_vasp: bool = False
    """Whether to generate VASP DFT input files for generated structures
    Creates INCAR, KPOINTS, POTCAR files for DFT verification workflow
    """

    # ==================== Miscellaneous ====================

    seed: int = 42
    """Random seed for reproducibility (affects PyTorch, NumPy, Python RNGs)""" 
