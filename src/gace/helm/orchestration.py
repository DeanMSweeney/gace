import contextlib
import csv
import json
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

from tqdm import tqdm

from gace.utils.path import Flow
from gace.utils.relaxation import Silent
from fairchem.core import OCPCalculator
from gace.utils.tft.tft_wrapper import PyGToTensorDict
from gace.utils.builder import sample_prim_atom_counts
from gace.utils.catalyst_forge import CatalystForge
import ase.io

from typing import Optional, Literal, Union, Tuple


class LightFlowMatch(LightningModule, Flow):
    """
    PyTorch Lightning module for training conditional flow matching models.

    Combines LightningModule (for training infrastructure) with Flow (for
    interpolation paths and loss computation). Supports the "forge"
    variant for de novo atomic structure generation.

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
            variant: str="forge",
            mode: str="debug",
            track_grad_norms: bool=False,
            generated_strucs=None,
            relaxed_strucs=None,
            sanitized_strucs=None,
            config_vasp: bool=False,
            x_matching: Literal["flow", "endpoint"]="flow",
            schedule_type: Literal["linear", "cosine"]="linear",
            optimal_transport: bool=False,
            informed_priors: bool=True,
            num_integration_steps: int=1,
            relax_otf: bool=False,
            batched_relaxation: bool=True,
            mlff_checkpoint: Optional[str]=None,
            accelerator: str="gpu",
            seed: int=42,
            num_prim_atoms: Optional[int]=None,
            target_binding_energy: Optional[float]=None,
            num_generated_samples: int=1,
            catalyst_dist_cutoff: float=0.75,
            ads_integrity_threshold: float=2.0,
            cfg_dropout_prob: float=0.1,
            cfg_w_continuous: float=1.0,
            cfg_w_discrete: float=1.0,
            sde_noise_scale: float=0.0,
            sanitize_otf: bool=True,
            **kwargs,
        ):

        super().__init__()
        # Save hyperparameters for logging and checkpointing (ignoring model-specific args)
        self.save_hyperparameters(ignore=["model", "variant", "mode", "rtol", "atol", "ode_method", \
            "track_grad_norms", "generated_strucs", "relaxed_strucs", "sanitized_strucs", "config_vasp", "threshold", "x_matching", \
            "interpolant_schedule", "write_eval_metrics", "num_integration_steps", "relax_otf", "mlff_checkpoint", \
            "accelerator", "seed", "num_prim_atoms", "optimal_transport"])

        # Model and training configuration
        self.model = model  # Neural network (e.g., Equiformer)
        self.variant = variant  # "refine" or "forge"
        self.mode = mode
        self.loss_fn = nn.MSELoss()  # Base loss function (not directly used, see criterion)

        # ODE solver configuration (for inference/testing)
        self.num_integration_steps = num_integration_steps

        # Monitoring and output configuration
        self.track_grad_norms = track_grad_norms
        self.generated_strucs = generated_strucs
        self.relaxed_strucs = relaxed_strucs
        self.sanitized_strucs = sanitized_strucs

        self.config_vasp = config_vasp
        self.catalyst_dist_cutoff = catalyst_dist_cutoff

        # Flow matching configuration
        self.x_matching = x_matching  # "flow" or "endpoint"
        self.schedule_type = schedule_type  # "linear" or "cosine"
        self.optimal_transport = optimal_transport
        self.informed_priors = informed_priors
        self.relax_otf = relax_otf
        self.sanitize_otf = sanitize_otf
        self.batched_relaxation = batched_relaxation
        self.accelerator = accelerator
        self.seed = seed
        self.num_prim_atoms = num_prim_atoms
        self.mlff_checkpoint = mlff_checkpoint
        self.ads_integrity_threshold = ads_integrity_threshold

        self._mlff_calculator = None
        if relax_otf and mlff_checkpoint is not None:
            with Silent():
                self._mlff_calculator = OCPCalculator(
                    model_name=mlff_checkpoint,
                    local_cache="./mlff_checkpoint_cache",
                    cpu=(accelerator == "cpu"),
                    seed=seed,
                )

        self.target_binding_energy = target_binding_energy
        self.num_generated_samples = num_generated_samples
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_w_continuous = cfg_w_continuous
        self.cfg_w_discrete = cfg_w_discrete
        self.sde_noise_scale = sde_noise_scale

        _hist_path = os.path.join(os.path.dirname(__file__), "primitive_atom_distribution.json")
        with open(_hist_path) as f:
            hist = json.load(f)["prim_slab_num_atoms_hist"]
        self.register_buffer("prim_atoms_hist", torch.tensor(hist, dtype=torch.float32))
       # ==============================================================================
       
    def _build_target_props(self, f_prim) -> dict:
        """Build the target-property conditioning dict for the model forward pass.

        Always passes binding energy. During training, randomly drops a fraction
        of samples (cfg_dropout_prob) by injecting a boolean mask; the model
        replaces those samples' embeddings with the learnable null embedding.
        """
        feats = {"ads_binding_energy": f_prim.data["ads_binding_energy"]}
        if self.training and self.cfg_dropout_prob > 0:
            B = feats["ads_binding_energy"].shape[0]
            feats["cfg_drop_mask"] = (
                torch.rand(B, device=feats["ads_binding_energy"].device) < self.cfg_dropout_prob
            )
        return feats

    def forward(self, data: Data, force_unconditional: bool = False) -> dict:
        """Run the forge model forward pass and return a flat prediction dict.

        Converts the PyG batch to dense tensors, calls the transformer, and
        unpacks outputs back to sparse form. Used both during training
        (via _transport_loss) and sampling (via Flow.sample),
        which calls this as `self(data)` because
        LightFlowMatch inherits from Flow.

        Args:
            data: PyG Data/Batch with all forge fields populated.
            force_unconditional: If True, drop all conditioning (cfg_drop_mask=True for
                every sample). Used for the unconditional forward pass in CFG guidance.

        Returns:
            Dict with keys: pos, z, len, ang, M, ads_pos.
        """
        wrapper = PyGToTensorDict()
        x_prim, x_ads, t_prim, t_ads, f_prim, f_ads = wrapper.to_tensordict(data)

        prim_padding_mask = ~x_prim.mask  # (B, M)
        ads_padding_mask  = ~x_ads.mask   # (B, N)

        t = t_prim.data["t"].squeeze(-1)  # (B,)

        if force_unconditional:
            B = x_prim.data["atomic_numbers"].shape[0]
            feats = {
                "ads_binding_energy": f_prim.data["ads_binding_energy"],
                "cfg_drop_mask": torch.ones(B, dtype=torch.bool, device=data.pos.device),
            }
        else:
            feats = self._build_target_props(f_prim)

        pred_prim, pred_ads  = self.model(
            x_prim=(
                x_prim.data["atomic_numbers"],          # (B, M) long
                x_prim.data["frac_coords"],            # (B, M, 3)
                x_prim.data["lengths_scaled"],         # (B, 1, 3)
                x_prim.data["angles_radians"],         # (B, 1, 3)
                x_prim.data["transformation_matrix"],  # (B, 3, 3)
                x_prim.data["supercell_lattice"],      # (B, 3, 3)
            ),
            x_ads=(
                x_ads.data["ads_atomic_numbers"],      # (B, N)
                x_ads.data["ads_pos"],                 # (B, N, 3)
                x_ads.data["ads_ref_centered_pos"],    # (B, N, 3)
            ),
            t_prim=(t, t, t, t),
            t_ads=(t,),
            feats=feats,
            prim_padding_mask=prim_padding_mask,
            ads_padding_mask=ads_padding_mask,
        )

        return {
            "pos":     x_prim.to_sparse(pred_prim[1]),   # (N_prim, 3)
            "z":       x_prim.to_sparse(pred_prim[0]),   # (N_prim, 100)
            "len":     pred_prim[2].squeeze(1),          # (B, 3)
            "ang":     pred_prim[3].squeeze(1),          # (B, 3)
            "M":       pred_prim[4],                     # (B, 3, 3)
            "ads_pos": x_ads.to_sparse(pred_ads[0]),     # (N_ads, 3)
        }

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
        # Generate interpolated training data at random time t.
        # path() stores supervision targets on the returned data object.
        data = self.build_path(input_graph=input_graph) 

        # Model predicts velocity field (or dict for forge with z_logits)
        if self.variant == "forge":
            pred = self(data)
            losses = self.criterion(input_graph=data, pred=pred)
        else:
            raise RuntimeError(f"Unknown variant: {self.variant}")

        return losses
    
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
        losses = self._transport_loss(data)

        for name, val in losses.items():
            self.log(
                f"train/{name}",
                val.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=(name == "total_loss"),
                batch_size=data.num_graphs,
                sync_dist=True,
            )

        return losses["total_loss"]
        
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
        losses = self._transport_loss(data)

        for name, val in losses.items():
            self.log(
                f"val/{name}",
                val.detach(),
                prog_bar=(name == "total_loss"),
                batch_size=data.num_graphs,
                sync_dist=True,
            )

        return losses["total_loss"]
    
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
        
        t = torch.linspace(0., 1, self.num_integration_steps, device=data.pos.device, dtype=data.pos.dtype)
        B = int(data.batch.max().item()) + 1
        counts = sample_prim_atom_counts(B, data.pos.device, self.prim_atoms_hist, self.num_prim_atoms)
        
        if self.target_binding_energy:
            data.ads_binding_energy = torch.ones(B, device=data.pos.device, dtype=data.pos.dtype) * self.target_binding_energy

        opt_graph = self.sample(input_graph=data, t=t, num_prim_atoms=counts,)

        B = int(opt_graph.batch.max().item()) + 1
        binding_energies = [data.ads_binding_energy[b].item() for b in range(B)]

        cf = CatalystForge(
            opt_graph=opt_graph,
            binding_energies=binding_energies,
            batch_idx=batch_idx,
            calculator=self._mlff_calculator,
            relax_otf=self.relax_otf,
            catalyst_dist_cutoff=self.catalyst_dist_cutoff,
            generated_strucs=self.generated_strucs,
            relaxed_strucs=self.relaxed_strucs,
            sanitized_strucs=self.sanitized_strucs,
            file_ext=".traj",
            sanitize_otf=self.sanitize_otf,
        )
        for metrics in cf.fire():
            for key, val in metrics.items():
                if isinstance(val, float):
                    self.log(key, val, on_step=False, on_epoch=True, batch_size=1)
        return

    def predict_step(self, data: Data, batch_idx: int):
        """Generate catalyst structures conditioned on adsorbate molecule and target binding energy.

        Called by trainer.predict() in generate mode.  Runs the forge Euler sampling
        loop and writes the resulting structures to gen_POS if set.

        Args:
            data: PyG Data built by build_molecule_data() — adsorbate conditioning
                  fields populated, catalyst fields are placeholder zeros.
            batch_idx: Index of this batch.

        Returns:
            List of ASE Atoms objects (one per graph in the batch).
        """
        if self.target_binding_energy is None:
            raise ValueError("No target binding energy provided.")

        B = data.num_graphs
        data.ads_binding_energy = torch.full(
            (B,), self.target_binding_energy, device=data.pos.device, dtype=torch.float32,
        )

        t = torch.linspace(0., 1., self.num_integration_steps, device=data.pos.device, dtype=torch.float32)
        counts = sample_prim_atom_counts(B, data.pos.device, self.prim_atoms_hist, self.num_prim_atoms)
        opt_graph = self.sample(input_graph=data, t=t, num_prim_atoms=counts)

        binding_energies = [data.ads_binding_energy[b].item() for b in range(B)]
        cf = CatalystForge(
            opt_graph=opt_graph,
            binding_energies=binding_energies,
            batch_idx=batch_idx,
            calculator=self._mlff_calculator,
            relax_otf=self.relax_otf,
            catalyst_dist_cutoff=self.catalyst_dist_cutoff,
            generated_strucs=self.generated_strucs,
            relaxed_strucs=self.relaxed_strucs,
            sanitized_strucs=self.sanitized_strucs,
            file_ext=".traj",
            sanitize_otf=self.sanitize_otf,
        )
        all_metrics = list(cf.fire())
        all_atoms = cf._all_atoms

        if all_metrics:
            keys = all_metrics[0].keys()
            avgs = {k: np.mean([m[k] for m in all_metrics if isinstance(m.get(k), float)]) for k in keys
                    if any(isinstance(m.get(k), float) for m in all_metrics)}
            print(f"[predict batch={batch_idx} avg over {len(all_metrics)} samples] " +
                  " ".join(f"{k}={v:.4f}" for k, v in avgs.items()))

        return all_atoms

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
        # None → constant learning rate, return early without a scheduler
        if self.hparams.lr_scheduler is None:
            return {"optimizer": optimizer}

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
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler!r}")

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
