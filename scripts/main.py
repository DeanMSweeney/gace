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

from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.utils.flow_matching import OptTransportMixin
from adsflow.utils.lmdb_dataset import LazyLmdbDataset
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from torchdiffeq import odeint

from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms


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

    # Model run mode
    variant:                         Literal["refine", "forge"] = "refine"
    mode:                            Literal["debug", "train", "eval", "generate"] = "train"

    # Sampler settings (ODEINT)
    rtol:                            float           = 1e-7
    atol:                            float           = 1e-9
    ode_method:                      str             = "dopri5"

    # Misc
    seed:                            int            = 42
    mask_atoms:                      bool           = True 
    


class LightFlowMatch(pl.LightningModule, OptTransportMixin):
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
            variant: str = "refine",
            mask_atoms: bool = True,
            lr: int = 1e-3, 
            rtol: float = 1e-7, 
            atol: float = 1e-9,
            ode_method: str = "dopri5",
            track_grad_norms: bool = False,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore="model")  # all __init__ args saved as hyperparameters
        self.model = model
        self.loss_fn = nn.MSELoss() # mean squared error loss function 
        self.variant = variant
        self.mask_atoms = mask_atoms
        self.rtol = rtol
        self.atol = atol 
        self.ode_method = ode_method
        self._prev_loss = None
        self.track_grad_norms = track_grad_norms


    def forward(self, data):
        return self.model(data)
    
    def _transport_loss(self, input_data: Data) -> torch.Tensor: 

        if self.global_step > 0: 
            self._prev_loss = self._loss.item()
            
        data, u = self.path(input_data, naive=True) 
        u_hat = self(data) # N, 3

        if self.mask_atoms:
            free_atoms = (1 - data.fixed).unsqueeze(1) # N
            loss = self.masked_loss(u_hat=u_hat, u=u, w=free_atoms) 
        else: 
            loss = self.loss_fn(u_hat, u)
        
        self._loss = loss
        
        if self._prev_loss is not None and loss > 10 * self._prev_loss and self.loss > 100:
            self.print(
                 f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [SPIKE] \n"
                 f"Current loss: {loss.item()} \n" 
                 f"Previous loss: {self._prev_loss} \n"
                 f"Predicted u: {torch.tensor(u_hat).abs().max()} \n" 
                 f"Actual u: {u.abs().max()} \n" 
                 f"t: {data.t.max()} \n"
            )
            
            
            x0 = Atoms(
                    positions=data.pos_init.detach().cpu().numpy(), # initial
                    numbers=data.atomic_numbers.detach().cpu().numpy(),
                    cell=data.cell.squeeze(0).detach().cpu().numpy(),
                    pbc=True,  # or (True,True,False) for a slab
                )
            x1 = Atoms(
                    positions=data.pos_relaxed.detach().cpu().numpy(), # initial
                    numbers=data.atomic_numbers.detach().cpu().numpy(),
                    cell=data.cell.squeeze(0).detach().cpu().numpy(),
                    pbc=True,  # or (True,True,False) for a slab
                )
            xt = Atoms(
                    positions=data.pos.detach().cpu().numpy(), # initial
                    numbers=data.atomic_numbers.detach().cpu().numpy(),
                    cell=data.cell.squeeze(0).detach().cpu().numpy(),
                    pbc=True,  # or (True,True,False) for a slab
                )
        
            p = "str"
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> WRITING TO POSCAR')
            write("POSCAR0",  x0, direct=False)  # Cartesian
            write("POSCAR1",  x1, direct=False)  # Cartesian
            write("POSCART",  xt, direct=False)  # Cartesian
            
            if self.global_step > 1000:
                l = p + 1
            
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
            return u 
        
        # define time interval for the IVP
        t = torch.linspace(0., 1., 100, 
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
            "train/loss",
            loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        return loss
    
    def validation_step(self, data: Data, batch_idx):
        loss = self._transport_loss(data)

        self.log(
            "val/loss", 
            loss, 
            prog_bar=True, 
            batch_size=len(data.natoms),
            sync_dist=True,
        )
        
        return loss

    def test_step(self, data: Data, batch_idx):
        loss = self._transport_loss(data)
        opt_graph, struc_rmse, null_struc_rmse, base_struc_rmse, velocity_loss = self._sampler(data)

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
        
    def on_before_optimize_step(self, optimizer): 
        if self.track_grad_norms:
                # Compute total gradient norm without clipping
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float('inf')
                    )
                # Log the gradient norm for monitoring
                self.log("unclipped grad norm", grad_norm)
        
        
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
        mask_atoms=cfg.mask_atoms,
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
        track_grad_norms=cfg.track_grad_norms,
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
    
   
