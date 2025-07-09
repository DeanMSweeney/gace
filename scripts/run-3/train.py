from dataclasses import dataclass

import torch
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from hydra import main
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.utils.data_handling import OCP_DataHandler

@dataclass
class TrainConfig:
    # model 
    model_cfg: DictConfig
    # seed  
    seed: int = 42 
    # paths
    oc20_train: str = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"
    oc20_val: str = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/val_id/data.lmdb"
    # training hyperparameters
    lr: float = 1e-4
    batch_size: int = 16
    num_workers: int = 4
    precision: str = "16-mixed"
    max_epochs: int = 300
    accumulate_grad_batches: int = 4
    accelerator: str = "gpu"
    devices: int = 1
    from_noise: bool = False
    # checkpoint & logging
    project: str = "adsflow"
    run_name: str = "gauntlet"
    ckpt_dir: str = "checkpoints"
    ckpt_monitor: str = "val/loss"
    ckpt_mode: str = "min"
    ckpt_save_top_k: int = 3
    ckpt_save_last: bool = True
    ckpt_path: str = None

class LightFlowMatch(pl.LightningModule):
    """
    Functionality 
    -------------
    This class defines the training algorithm for the conditional flow matching model using optimal transport 
    
    Arguments
    ---------
    model: dict 
        Model configuration 
    lr: int
        Learning rate 
    from_noise: bool 
        Whether to learn mapping from Gaussian noise or an initial structure
    """
    def __init__(
            self, 
            model: GemNetFlow,
            lr: int = 1e-3, 
            from_noise: bool = False
        ):
        
        super().__init__()
        self.save_hyperparameters()  # all __init__ args saved
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.from_noise = from_noise

    def forward(self, data):
        return self.model(data)
    
    def _compute_loss(self, data: Data) -> torch.Tensor: 
        x1 = data.pos_relaxed.clone() # sample relaxed positions and discard
        del data.pos_relaxed

        if self.from_noise:
            x0 = torch.randn_like(x1) # sample noise 
        else:
            x0 = data.pos_init.clone() # sample initial positions and discard 
            del data.pos_init

        t_graph = torch.rand(len(data.natoms), 1, device=x1.device) # sample t from U
        t = t_graph[data.batch]

        xt = t * x1 + (1 - t) * x0 # interpolation scheme 
        data.pos = xt # reassign xt to pos 
        data.t = t

        u_hat = self(data) # compute velocity field 
        u = x1 - x0
        return self.loss_fn(u_hat, u)

    # define training and validation methods 
    def training_step(self, data: Data, batch_idx):
        loss = self._compute_loss(data)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_step", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, data: Data, batch_idx):
        loss = self._compute_loss(data)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer)
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler1, scheduler2]

@main(config_path="conf", config_name="train_config")
def main(cfg: TrainConfig) -> None:
    # seed for reproducibility 
    pl.seed_everything(cfg.seed, workers=True)

    # build model with configuration file
    model: GemNetFlow = instantiate(cfg.model_cfg, _convert_="partial")

    # load datasets
    pin_memory = True if cfg.accelerator == "gpu" else False
    train_set = OCP_DataHandler(cfg.oc20_train, pbc=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=pin_memory)
    val_set = OCP_DataHandler(cfg.oc20_val, pbc=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, pin_memory=pin_memory)

    lightning = LightFlowMatch(model, lr=cfg.lr, from_noise=cfg.from_noise)

    wandb_logger = WandbLogger(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model="all",
    )
    wandb_logger.watch(lightning.model, log="all", log_freq=50)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=f"{cfg.run_name}-{{epoch:02d}}",
        monitor=cfg.ckpt_monitor,
        mode=cfg.ckpt_mode,
        save_top_k=cfg.ckpt_save_top_k,
        save_last=cfg.ckpt_save_last,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy="ddp_find_unused_parameters_true",
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=10,
    )

    trainer.fit(
        lightning,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path, # load model from checkpoint
    )


if __name__ == "__main__":
    main()
