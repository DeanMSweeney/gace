import os
import numpy as np
import torch
from torch import optim 
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, PolynomialLR
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch_geometric.data import Data
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from adsflow.utils.x_matching import XMatching
from adsflow.utils.post_processing import analyze_loss

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
            variant: str="refine",
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
            **kwargs,
        ):

        super().__init__()
        # Save hyperparameters for logging and checkpointing (ignoring model-specific args)
        self.save_hyperparameters(ignore=["model", "variant", "mode", "rtol", "atol", "ode_method", "track_grad_norms", "in_POS", "gen_POS", "rlxd_POS", "config_vasp", "threshold", "x_matching", "interpolant_schedule"])

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

        # =============================================================================
        # Diagnostic mode: detailed loss analysis for model development
        # When enabled, collects per-atom statistics for problematic predictions
        self.diagnostics = False
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

    def _transport_loss(self, input_data: Data) -> torch.Tensor:
        """
        Compute the flow matching loss for a batch of structures.

        Steps:
        1. Sample random interpolation time t for each structure
        2. Interpolate positions: x(t) = x0 + alpha(t) * (x1 - x0)
        3. Predict velocity field: f = model(x(t), t)
        4. Compute loss between predicted and target velocity

        Args:
            input_data: Batch of atomic structures

        Returns:
            loss: Average loss over batch
            max_loss: Maximum per-atom loss (for monitoring outliers)
            max_loss_idx: Index of atom with maximum loss
        """
        # Generate interpolated training data at random time t
        # data.pos = x(t), u = target velocity dx/dt
        data, u = self.path(input_data=input_data) 

        # Model predicts velocity field (or positions if endpoint matching)
        f = self(data)

        if self.variant == "refine":
            # Compute loss between predicted and target
            # For flow matching: loss = |f - u|^2 (velocity matching)
            # For endpoint matching: loss = |f - x1|^2 (position matching)
            loss, max_loss, max_loss_idx = self.criterion(
                input_data=data,
                a=f,  # Model prediction
                b=data.pos_relaxed if self.x_matching == "endpoint" else u,  # Target
                tao=self.hparams.tao,  # Adsorbate weighting factor
            )

        return loss, max_loss, max_loss_idx
    
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
            loss,
            on_step=True,  # Log every step
            on_epoch=True,  # Also compute epoch average
            prog_bar=True,  # Show in progress bar
            batch_size=len(data.natoms),  # For proper averaging
            sync_dist=True,  # Synchronize across GPUs
        )

        # Log maximum per-atom loss (for monitoring outliers)
        self.log(
            "train/max_loss",
            max_loss,
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
        if hasattr(data, "Fmax"):
            data.dE = torch.zeros_like(data.Fmax)

        loss, max_loss, _ = self._transport_loss(data)

        # Log validation metrics (no on_step to avoid clutter)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "val/max_loss",
            max_loss,
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
        2. Generate relaxed structure via ODE integration (if not in diagnostics mode)
        3. Compute evaluation metrics (MAE, DwT, ADwT)
        4. Optionally save structures to files

        Args:
            data: Batch of atomic structures
            batch_idx: Index of batch in test set

        Returns:
            loss: Test loss for this batch
        """
        # Compute training-style loss for comparison
        loss, max_loss, max_loss_idx = self._transport_loss(data)

        data.dE = torch.zeros_like(data.y_relaxed)
        data.Fmax = torch.ones_like(data.y_relaxed) * 0.02

        if self.diagnostics:
            # Diagnostic mode: detailed loss analysis for problematic samples
            if loss > 0.001:
                self.bad_iter += 1
                # Collect detailed statistics about the loss
                (self.DS0,  # displacement from adsorbate
                self.DS1,   # atomic numbers at high loss atoms
                self.DS2,   # tags at high loss atoms
                self.DS3,   # atomic numbers by distance
                self.DS4,   # tags by distance
                self.DS5,   # max distance
                self.DS6,   # max loss value
                self.DS7) = analyze_loss(  # average distance
                    data=data,
                    loss=loss,
                    max_loss=max_loss,
                    max_loss_idx=max_loss_idx,
                    bad_iter=self.bad_iter,
                )
            # Skip ODE integration in diagnostics mode
            (
                opt_graph,
                generated_mae,
                generated_dwt,
                generated_adwt,
                initial_mae,
                initial_dwt,
                initial_adwt,
            ) = 0, 0, 0, 0, 0, 0, 0

        else:
            # Normal test mode: generate structures via ODE integration
            # Define time interval for IVP (Initial Value Problem): t ∈ [0, 1]
            # Use 100 time steps for accurate integration
            t = torch.linspace(0., 1., 100, device=data.pos.device, dtype=data.pos.dtype)

            # Generate relaxed structure and compute metrics
            (
                opt_graph,          # Optimized structure data
                generated_mae,      # Mean Absolute Error (generated)
                generated_dwt,      # Distance within Threshold (generated)
                generated_adwt,     # Adsorbate Distance within Threshold (generated)
                initial_mae,        # MAE for initial structure (baseline)
                initial_dwt,        # DwT for initial structure
                initial_adwt,       # ADwT for initial structure
            ) = self.sample(input_data=data, t=t, threshold=self.threshold)

        # ==================== Log Test Metrics ====================

        # Loss metrics (training-style loss for comparison)
        self.log(
            "test/loss",
            loss,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "test/max_loss",
            max_loss,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        # Structural metrics: Mean Absolute Error (Angstroms)
        self.log(
            "test/generated_mae",  # Error of generated structure
            generated_mae,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "test/initial_mae",  # Error of initial structure (baseline)
            initial_mae,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        # Distance within Threshold (% of atoms within threshold distance)
        self.log(
            "test/generated_dwt",
            generated_dwt,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "test/initial_dwt",
            initial_dwt,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        # Adsorbate Distance within Threshold (ADwT metric, adsorbate-specific)
        self.log(
            "test/generated_adwt",
            generated_adwt,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        self.log(
            "test/initial_adwt",
            initial_adwt,
            prog_bar=True,
            batch_size=len(data.natoms),
            sync_dist=True,
        )

        # Ensure output directory exists for saving structures
        out_dir = self.trainer.default_root_dir
        os.makedirs(out_dir, exist_ok=True)

        return loss    

    def on_test_end(self):
        """
        Called after all test batches complete.

        If diagnostics mode is enabled, saves detailed loss analysis to .npy files.
        These files contain per-atom statistics for analyzing model failures.
        """
        if self.diagnostics:
            print(">>> You are running with diagnostic mode turned on ‼️")

            # Convert Python lists → tensors for multi-GPU gathering
            DS0 = torch.tensor(self.DS0, dtype=torch.float32)  # displacement from adsorbate
            DS1 = torch.tensor(self.DS1, dtype=torch.long)     # atomic numbers (high loss)
            DS2 = torch.tensor(self.DS2, dtype=torch.long)     # tags (high loss)
            DS3 = torch.tensor(self.DS3, dtype=torch.long)     # atomic numbers (by distance)
            DS4 = torch.tensor(self.DS4, dtype=torch.long)     # tags (by distance)
            DS5 = torch.tensor(self.DS5, dtype=torch.float32)  # max distance
            DS6 = torch.tensor(self.DS6, dtype=torch.float32)  # max loss value
            DS7 = torch.tensor(self.DS7, dtype=torch.float32)  # average distance

            # Helper to convert tensors to 1D NumPy arrays
            def unwrap(x):
                return x.reshape(-1).cpu().numpy()

            # Save diagnostic data for post-hoc analysis
            np.save("loss-displacement_from_ads", unwrap(DS0))  # loss vs distance from adsorbate
            np.save("loss-atomic_number",        unwrap(DS1))   # loss vs atomic number
            np.save("loss-tag",                  unwrap(DS2))   # loss vs atom tag
            np.save("distance-atomic_number",    unwrap(DS3))   # distance error vs atomic number
            np.save("distance-tag",              unwrap(DS4))   # distance error vs atom tag
            np.save("distance-max",              unwrap(DS5))   # maximum distance error
            np.save("distance-max_loss",         unwrap(DS6))   # maximum loss value
            np.save("avg_distance",              unwrap(DS7))   # average distance error


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


