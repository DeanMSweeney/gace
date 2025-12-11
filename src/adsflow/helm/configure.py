from omegaconf import OmegaConf, DictConfig, MISSING
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Union, Tuple


@dataclass 
class OptimizerConfig:
    
    # Training hyperparameters/settings
    lr:                              float = 1e-3
    weight_decay:                    float   = 0.0001
    betas:                           Tuple[float, float]=(0.9, 0.99)
    eps:                             float=1e-8
    lr_scheduler:                    Literal["polynomial_decay", "cosine_w_warmup", "cosine_annealing", "polynomial_w_warmup", "polynomial_decay"] = "cosine_w_warmup"
    num_cycles:                      float=0.50
    warmup_proportion:               float=0.01
    eta_min:                         float=0.0
    power:                           float=1.0
    max_epochs:                      int   = 100
    alpha:                           float = 0 
    
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
    optimizer_hparams: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Dataset paths
    oc20_train:                      Optional[str] = None
    oc20_val:                        Optional[str] = None
    oc20_test:                       Optional[str] = None
    oc20_mini:                       Optional[str] = None

    # DataLoader settings
    batch_size:                      int = 1
    num_workers:                     Optional[int] = None
    accumulate_grad_batches:         int   = 1
    val_check_interval:              int   = 1.0

    # Compute architecture
    accelerator:                     str = "gpu"
    devices:                         int = 1
    num_nodes:                       int = 1
    strategy:                        str = "auto"
    precision:                       Literal[16, "16-mixed", "bf16-mixed", 32] = "bf16-mixed"

    # Logging
    logger:                          bool                    = False
    project:                         str                     = "NAME_PROJ"
    run_name:                        str                     = "NAME_RUN"
    run_id:                          str                     = None
    track_grad_norms:                bool                    = True
    
    # Model Checkpointing
    ckpt_dir:                        str                     = "checkpoints"
    
    # Restart model from checkpoint
    ckpt_path:                       Optional[str]           = None
    resume_training:                      bool               = True

    # Model run mode
    variant:                         Literal["refine", "forge"] = "refine"
    mode:                            Literal["debug", "train", "eval", "generate"] = "generate"
    regressor:                       Literal["GemNet-OC", "EquiformerV2"] = "GemNet-OC"

    # Sampler settings (ODEINT)
    rtol:                            float           = 1e-7
    atol:                            float           = 1e-9
    ode_method:                      str             = "dopri5"
    
    # Generation 
    in_POS:                         Optional[str]         = None
    gen_POS:                    Optional[str]             = None
    rlxd_POS:                   Optional[str]             = None
    config_vasp:                       bool               = False 

    # Misc
    seed:                            int            = 42
    mask_frzn_atoms:                 bool           = True 
