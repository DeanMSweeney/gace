import contextlib
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, PolynomialLR
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch_geometric.data import Data
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from adsflow.utils.x_matching import XMatching
from adsflow.utils.evaluation import compute_eval_metrics
from adsflow.utils.relaxation import relax_struc
from adsflow.utils.atoms import write_strucs

from typing import Optional, Literal, Union, Tuple


class LightFlowMatch(LightningModule, XMatching):
    """
    PyTorch Lightning module for training conditional flow matching models.

    Combines LightningModule (for training infrastructure) with XMatching (for
    interpolation paths and loss computation). Supports both "refine" and "forge"
    variants for atomic structure relaxation.

    Args
    ---------
    model: nn.Module
        The neural network model (e.g., Equiformer) that predicts velocity fields
    variant: str
        Model variant: "refine" (from initial to relaxed) or "forge" (from noise)
    mode: str
        Training mode (e.g., "debug" for development)
    mask_frzn_atoms: bool
        Whether to mask frozen atoms in loss computation
    rtol: float
        Relative tolerance for ODE solver during inference
    atol: float
        Absolute tolerance for ODE solver during inference
    ode_method: str
        ODE integration method (e.g., "dopri5", "euler", "rk4")
    track_grad_norms: bool
        Whether to log gradient norms for monitoring training stability
    in_POS: str
        Path to save initial structure files during testing
    gen_POS: str
        Path to save generated structure files during testing
    rlxd_POS: str
        Path to save relaxed (ground truth) structure files during testing
    config_vasp: bool
        Whether to configure VASP DFT input files during testing
    threshold: float
        Distance threshold (Angstroms) for computing DwT/ADwT metrics
    x_matching: Literal["flow", "endpoint"]
        Loss computation mode: "flow" (velocity matching) or "endpoint" (position matching)
    interpolant_schedule: Literal["linear", "cosine"]
        Interpolation schedule: "linear" or "cosine" (slower at ends)
    **kwargs:
        Additional hyperparameters (lr, betas, weight_decay, etc.)
    """
    def __init__(
            self,
            model,
            variant: str="relax",
            mode: str="debug",
            rtol: float = 1e-7,
            atol: float = 1e-9,
            ode_method: str = "dopri5",
            track_grad_norms: bool=False,
            in_POS=None,
            gen_POS=None,
            rlxd_POS=None,
            config_vasp: bool=False,
            threshold: float=0.20,
            x_matching: Literal["flow", "endpoint"]="flow",
            schedule_type: Literal["linear", "cosine"]="linear",
            write_eval_metrics: bool=False,
            num_integration_steps: int=1,
            run_mlff: bool=False,
            mlff_checkpoint: Optional[str]=None,
            accelerator: str="gpu",
            seed: int=42,
            **kwargs,
        ):

        super().__init__()
        # Save hyperparameters for logging and checkpointing (ignoring model-specific args)
        self.save_hyperparameters(ignore=["model", "variant", "mode", "rtol", "atol", "ode_method", \
            "track_grad_norms", "in_POS", "gen_POS", "rlxd_POS", "config_vasp", "threshold", "x_matching", \
            "interpolant_schedule", "write_eval_metrics", "num_integration_steps", "run_mlff", "mlff_checkpoint", \
            "accelerator", "seed"])

        # Model and training configuration
        self.model = model  # Neural network (e.g., Equiformer)
        self.variant = variant  # "refine" or "forge"
        self.mode = mode
        self.loss_fn = nn.MSELoss()  # Base loss function (not directly used, see criterion)

        # ODE solver configuration (for inference/testing)
        self.rtol = rtol
        self.atol = atol
        self.ode_method = ode_method

        # Monitoring and output configuration
        self.track_grad_norms = track_grad_norms
        self.in_POS = in_POS
        self.gen_POS = gen_POS
        self.rlxd_POS = rlxd_POS
        self.config_vasp = config_vasp
        self.threshold = threshold

        # Flow matching configuration
        self.x_matching = x_matching  # "flow" or "endpoint"
        self.schedule_type = schedule_type  # "linear" or "cosine"
        self.write_eval_metrics = write_eval_metrics
        self.num_integration_steps = num_integration_steps
        self.run_mlff = run_mlff
        self.accelerator = accelerator
        self.seed = seed
        self.mlff_checkpoint = mlff_checkpoint

        # Diagnostic mode: detailed loss analysis for model development
        self._write_f_u = False
        self.bad_iter, self.good_ter, self.iter = 0, 0, 0
       # ==============================================================================
       
    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: Batch of atomic structures with positions, atomic numbers, etc.

        Returns:
            Predicted velocity field u(x, t) for flow matching
        """
        return self.model(data)

    def _transport_loss(self, input_graph: Data) -> torch.Tensor:
        """
        Compute the flow matching loss for a batch of structures.

        Steps:
        1. Sample random interpolation time t for each structure
        2. Interpolate positions: x(t) = x0 + alpha(t) * (x1 - x0)
        3. Predict velocity field: f = model(x(t), t)
        4. Compute loss between predicted and target velocity

        Args:
            input_graph: Batch of atomic structures

        Returns:
            loss: Average loss over batch
            max_loss: Maximum per-atom loss (for monitoring outliers)
            max_loss_idx: Index of atom with maximum loss
            velocity_predictions: Dict with t, predicted velocity, target velocity, batch, tags
        """
        # Generate interpolated training data at random time t
        # data.pos = x(t), u = target velocity dx/dt
        data, u = self.path(input_graph=input_graph) 

        # Model predicts velocity field (or positions if endpoint matching)
        f = self(data)
        if self.variant == "relax":
            # Compute loss between predicted and target
            # For flow matching: loss = |f - u|^2 (velocity matching)
            # For endpoint matching: loss = |f - x1|^2 (position matching)
            target = data.pos_relaxed if self.x_matching == "endpoint" else u
            loss, max_loss = self.criterion(
                input_graph=data,
                a=f,  # Model prediction
                b=target,  # Target
                loss_fn=self.hparams.loss_fn,
            )

        # DEBUG: collect diagnostic tensors
        if self._write_f_u: 
            velocity_predictions = {
                't': data.t.detach().cpu(),              # (N, 1) interpolation time
                'f': f.detach().cpu(),                    # (N, 3) predicted velocity
                'u': target.detach().cpu(),               # (N, 3) target velocity
                'batch': data.batch.detach().cpu(),        # (N,)   graph index
                'tags': data.tags.detach().cpu(),           # (N,)   atom tags
            }
        else:
            velocity_predictions = None

        return loss, max_loss, velocity_predictions
    
    # ==================== Training and Validation Methods ====================

    def training_step(self, data: Data, batch_idx):
        """
        Single training step (one batch).

        Args:
            data: Batch of atomic structures
            batch_idx: Index of batch in epoch

        Returns:
            loss: Loss value for backpropagation
        """
        loss, max_loss, _ = self._transport_loss(data)

        # Log average loss (weighted by batch size for multi-GPU training)
        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,  # Log every step
            on_epoch=True,  # Also compute epoch average
            prog_bar=True,  # Show in progress bar
            batch_size=len(data.natoms),  # For proper averaging
            sync_dist=True,  # Synchronize across GPUs
        )

        # Log maximum per-atom loss (for monitoring outliers)
        self.log(
            "train/max_loss",
            max_loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        return loss
        


    def validation_step(self, data: Data, batch_idx):
        """
        Single validation step (one batch).

        Computes loss on validation set without updating model parameters.
        Used for monitoring generalization and selecting best checkpoint.

        Args:
            data: Batch of atomic structures
            batch_idx: Index of batch in validation set

        Returns:
            loss: Validation loss for this batch
        """
        # Initialize energy difference if Fmax exists (for compatibility)
        data.Fmax = torch.ones_like(data.y_relaxed) * 0.03 
        data.dE = torch.zeros_like(data.Fmax)
        loss, max_loss, _ = self._transport_loss(data)

        # Log validation metrics (no on_step to avoid clutter)
        self.log(
            "val/loss",
            loss.detach(),
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "val/max_loss",
            max_loss.detach(),
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        return loss
    
    def test_step(self, data: Data, batch_idx):
        """
        Single test step (one batch).

        For each structure:
        1. Compute training loss (for comparison)
        2. Generate relaxed structure via ODE integration (if not in velocity_predictions mode)
        3. Compute evaluation metrics (MAE, DwT, ADwT)
        4. Optionally save structures to files

        Args:
            data: Batch of atomic structures
            batch_idx: Index of batch in test set

        Returns:
            loss: Test loss for this batch
        """
        # Compute training-style loss for comparison
        data.dE = torch.ones_like(data.y_relaxed) * 0.05
        data.Fmax = torch.ones_like(data.y_relaxed) * 0.10
        
        # DEBUG: collect velocity velocity_predictions for parquet output
        if self._write_f_u:
            loss, max_loss, velocity_predictions = self._transport_loss(data)
            if not hasattr(self, 'self._f_u'):
                self.self._f_u = []
            self.self._f_u.append(velocity_predictions)
        else:
            if self.num_integration_steps > 1:
                # Iterative self-refinement: re-apply model to its own output
                t = torch.linspace(0., 1, self.num_integration_steps, device=data.pos.device, dtype=data.pos.dtype)
                opt_graph = self.sample(input_graph=data, t=t)
            else:
                opt_graph = data 
                opt_graph.pos_generated = opt_graph.pos   
            
            opt_graph = relax_struc(
                opt_graph,
                run_mlff=self.run_mlff,
                config_vasp=self.config_vasp,
                mlff_checkpoint=self.mlff_checkpoint,
                accelerator=self.accelerator,
                seed=self.seed,
                gen_POS=self.gen_POS,
                in_POS=self.in_POS,
                rlxd_POS=self.rlxd_POS,
            )

            write_strucs(
                data=opt_graph,
                gen_POS=self.gen_POS,
                in_POS=self.in_POS,
                rlxd_POS=self.rlxd_POS,
            )

            eval_metrics = compute_eval_metrics(data=opt_graph, threshold=self.threshold)

            if self.write_eval_metrics:
                if not hasattr(self, "eval_metrics"):
                    self.eval_metrics = {k: [] for k in eval_metrics.keys()}
                for k, v in eval_metrics.items():
                    self.eval_metrics[k].append(v) # only append updated batch values

            # Log all evaluation metrics
            if eval_metrics:
                for name, value in eval_metrics.items():
                    self.log(f"test/{name}", value, prog_bar=True, batch_size=len(data.natoms), sync_dist=True)
        
        return   

    def on_test_end(self):
        """
        Called after all test batches complete.

        Saves evaluation metrics to CSV and velocity velocity_predictions to parquet.
        """
        # DEBUG: save velocity velocity_predictions to parquet
        if self._write_f_u and hasattr(self, 'self._f_u'):
            t = torch.cat([d['t'] for d in self.self._f_u], dim=0).squeeze(-1).numpy()
            f = torch.cat([d['f'] for d in self.self._f_u], dim=0).numpy()
            u = torch.cat([d['u'] for d in self.self._f_u], dim=0).numpy()
            tags = torch.cat([d['tags'] for d in self.self._f_u], dim=0).numpy()

            # Offset batch indices so they're globally unique across batches
            offset = 0
            batch_global = []
            for d in self.self._f_u:
                b = d['batch'].numpy()
                batch_global.append(b + offset)
                offset += b.max() + 1
            batch = np.concatenate(batch_global)

            df = pd.DataFrame({
                't': t,
                'f_x': f[:, 0], 'f_y': f[:, 1], 'f_z': f[:, 2],
                'u_x': u[:, 0], 'u_y': u[:, 1], 'u_z': u[:, 2],
                'batch': batch,
                'tags': tags,
            })

            output_path = './velocity_predictions.parquet'
            df.to_parquet(output_path, index=False)
            if self.global_rank == 0:
                print(f"\n[DEBUG] Saved velocity velocity_predictions to {output_path} ({len(df)} atoms)")

        if self.write_eval_metrics and hasattr(self, "eval_metrics"):
            output_dir = './eval_metrics'

            # Check if running distributed
            if self.trainer.world_size > 1:
                # Gather metrics from all GPUs
                gathered_metrics = {}
                for k, v in self.eval_metrics.items():
                    eval_tensor = torch.tensor(v, device=self.device)
                    gathered_eval_tensors = self.all_gather(eval_tensor)
                    gathered_metrics[k] = gathered_eval_tensors.flatten().cpu().tolist()

                # Only rank 0 writes the CSV
                if self.global_rank == 0:
                    os.makedirs(output_dir, exist_ok=True)
                    pd.DataFrame(gathered_metrics).to_csv(os.path.join(output_dir, 'eval_metrics.csv'), index=False)
            else:
                # Single GPU - write directly
                os.makedirs(output_dir, exist_ok=True)
                pd.DataFrame(self.eval_metrics).to_csv(os.path.join(output_dir, 'eval_metrics.csv'), index=False)

    # ==================== Optimizer Configuration ====================

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Optimizer: AdamW (Adam with decoupled weight decay)

        Supported LR Schedulers:
        - cosine_w_warmup: Cosine annealing with linear warmup (recommended)
        - cosine_annealing: Standard cosine annealing without warmup
        - polynomial_w_warmup: Polynomial decay with linear warmup
        - polynomial_decay: Polynomial decay without warmup

        Returns:
            Dictionary with optimizer and lr_scheduler configuration
        """
        # AdamW optimizer: Adam with improved weight decay handling
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,              # Base learning rate
            betas=self.hparams.betas,        # Adam beta parameters (momentum terms)
            weight_decay=self.hparams.weight_decay,  # L2 regularization
            eps=self.hparams.eps,            # Numerical stability epsilon
        )

        # Scheduler updates every step (not every epoch)
        interval = "step"

        # Calculate total training steps (batches × epochs)
        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_warmup_steps = int(self.hparams.warmup_proportion * num_training_steps)

        # Select and configure learning rate scheduler
        lr_scheduler = str(self.hparams.lr_scheduler)

        if lr_scheduler == "cosine_w_warmup":
            # Cosine annealing with warmup: lr rises linearly then follows cosine decay
            # Recommended for most training runs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,      # Linear warmup period
                num_training_steps=num_training_steps,  # Total training steps
                num_cycles=self.hparams.num_cycles      # Number of cosine cycles (usually 0.5)
            )

        elif lr_scheduler == "cosine_annealing":
            # Standard cosine annealing without warmup
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,        # Period of cosine schedule
                eta_min=1e-3*self.hparams.lr,    # Minimum learning rate (0.001 × base_lr)
            )

        elif lr_scheduler == "polynomial_w_warmup":
            # Polynomial decay with warmup: lr = base_lr × (1 - t/T)^power
            if self.hparams.num_decay_steps > 0:
                # Custom decay period
                anneal_total_steps = self.hparams.num_decay_steps + num_warmup_steps
            else:
                # Decay over entire training
                anneal_total_steps = num_training_steps
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=anneal_total_steps,
                lr_end=1e-3*self.hparams.lr,     # Final learning rate
                power=self.hparams.power         # Polynomial exponent (e.g., 1.0 for linear)
            )

        elif lr_scheduler == "polynomial_decay":
            # Polynomial decay without warmup
            scheduler = PolynomialLR(
                optimizer=optimizer,
                total_iters=num_training_steps,
                power=self.hparams.power,
            )
        else:
            raise ValueError(f"Scheduler not specified")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,    # Update every step
                "frequency": 1,          # Update every interval
            }
        }

    # ==================== Optimization Callbacks ====================

    def on_before_optimize_step(self, optimizer):
        """
        Called before the optimizer updates parameters.

        If gradient norm tracking is enabled, computes and logs the total gradient norm.
        Useful for monitoring training stability and detecting exploding gradients.

        Args:
            optimizer: The optimizer about to update parameters
        """
        if self.track_grad_norms:
                # Compute total gradient norm without clipping (max_norm=inf)
                # Returns: sqrt(sum(grad^2)) across all parameters
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float('inf')
                    )
                # Log for monitoring (large values indicate instability)
                self.log("unclipped grad norm", grad_norm)
