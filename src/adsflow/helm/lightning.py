import contextlib
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

from adsflow.utils.path import Flow
from adsflow.utils.relaxation import relax_struc
from adsflow.utils.atoms import write_strucs
from adsflow.utils.tft.tft_wrapper import PyGToTensorDict
from adsflow.utils.builder import forge_catalyst, sample_prim_atom_counts
from adsflow.utils.evaluator import check_validity
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
            optimal_transport: bool=False,
            informed_priors: bool=True,
            operate_on_torus: bool=False,
            write_eval_metrics: bool=False,
            num_integration_steps: int=1,
            run_mlff: bool=False,
            mlff_checkpoint: Optional[str]=None,
            accelerator: str="gpu",
            seed: int=42,
            num_prim_atoms: Optional[int]=None,
            refine_final: bool=False,
            target_binding_E: Optional[float]=None,
            num_generated_samples: int=1,
            covalent_radii_scale: float=0.75,
            condition_on_binding_E: bool=True,
            standardize_ads_pos: bool=False,
            ads_pos_mean: list = [0.02587131, -0.69517581, 0.91811687],
            ads_pos_std: list = [8.21663071, 9.02733422, 9.86882174],
            **kwargs,
        ):

        super().__init__()
        # Save hyperparameters for logging and checkpointing (ignoring model-specific args)
        self.save_hyperparameters(ignore=["model", "variant", "mode", "rtol", "atol", "ode_method", \
            "track_grad_norms", "in_POS", "gen_POS", "config_vasp", "threshold", "x_matching", \
            "interpolant_schedule", "write_eval_metrics", "num_integration_steps", "run_mlff", "mlff_checkpoint", \
            "accelerator", "seed", "num_prim_atoms", "optimal_transport"])

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
        self.covalent_radii_scale = covalent_radii_scale

        # Flow matching configuration
        self.x_matching = x_matching  # "flow" or "endpoint"
        self.schedule_type = schedule_type  # "linear" or "cosine"
        self.optimal_transport = optimal_transport
        self.informed_priors = informed_priors
        self.operate_on_torus = operate_on_torus
        self.write_eval_metrics = write_eval_metrics
        self.num_integration_steps = num_integration_steps
        self.run_mlff = run_mlff
        self.accelerator = accelerator
        self.seed = seed
        self.num_prim_atoms = num_prim_atoms
        self.refine_final = refine_final
        self.mlff_checkpoint = mlff_checkpoint
        self.target_binding_E = target_binding_E
        self.num_generated_samples = num_generated_samples
        self.condition_on_binding_E = condition_on_binding_E
        self.standardize_ads_pos = standardize_ads_pos

        # Adsorbate position normalization stats (per-axis), computed offline over training set.
        # Path, loss, and Euler integration run in normalized space; denormalize only at sampling output.
        self.register_buffer("ads_pos_mean", torch.tensor(ads_pos_mean, dtype=torch.float32))
        self.register_buffer("ads_pos_std", torch.tensor(ads_pos_std, dtype=torch.float32))

        _hist_path = os.path.join(os.path.dirname(__file__), "primitive_atom_distribution.json")
        with open(_hist_path) as f:
            hist = json.load(f)["prim_slab_num_atoms_hist"]
        self.register_buffer("prim_atoms_hist", torch.tensor(hist, dtype=torch.float32))

        # Diagnostic mode: detailed loss analysis for model development
        self._write_f_u = False
        self.bad_iter, self.good_ter, self.iter = 0, 0, 0
       # ==============================================================================
       
    def forward(self, data: Data) -> dict:
        """Run the forge model forward pass and return a flat prediction dict.

        Converts the PyG batch to dense tensors, calls the transformer, and
        unpacks outputs back to sparse form. Used both during training
        (via _transport_loss) and sampling (via Flow.sample),
        which calls this as `self(data)` because
        LightFlowMatch inherits from Flow.

        Args:
            data: PyG Data/Batch with all forge fields populated.

        Returns:
            Dict with keys: pos, z, len, ang, M, ads_pos.
        """
        wrapper = PyGToTensorDict()
        x_prim, x_ads, t_prim, t_ads, f_prim, f_ads = wrapper.to_tensordict(data)

        prim_padding_mask = ~x_prim.mask  # (B, M)
        ads_padding_mask  = ~x_ads.mask   # (B, N)

        t = t_prim.data["t"].squeeze(-1)  # (B,)

        pred_prim, pred_ads, _ = self.model(
            x_prim=(
                x_prim.data["atomic_numbers_hot"],     # (B, M, 118)
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
            feats={"ads_binding_energy": f_prim.data["ads_binding_energy"] if self.condition_on_binding_E else None},
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
        data = self.path(input_graph=input_graph) 

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
        opt_graph = self.sample(input_graph=data, t=t, num_prim_atoms=counts, refine_final=self.refine_final)

        prim_batch = opt_graph.batch[opt_graph.tags == 1]
        ads_batch  = opt_graph.batch[opt_graph.tags == 2]
        num_graphs = int(opt_graph.batch.max().item()) + 1

        for b in range(num_graphs):
            atoms = forge_catalyst(opt_graph, b, prim_batch, ads_batch)

            if atoms is not None: 
                atoms_rlxd = relax_struc(
                        atoms=atoms,
                        run_mlff=self.run_mlff,
                        config_vasp=self.config_vasp,
                        mlff_checkpoint=self.mlff_checkpoint,
                        accelerator=self.accelerator,
                        seed=self.seed,
                    )
                
                atoms_rlxd = atoms_rlxd if self.run_mlff else atoms
                target_ads_dists = opt_graph.ads_pairwise_dist[ads_batch == b].cpu().numpy()  # (N_ads_b, 15)
                result = check_validity(atoms_rlxd, self.covalent_radii_scale, target_ads_dists=target_ads_dists)

                metrics = {
                    "vol_ok":      float(result["vol_ok"]),
                    "cat_dist_ok":     float(result["cat_dist_ok"]),
                    "ads_dist_ok": float(result["ads_dist_ok"]),
                    "sample_is_valid":    float(result["is_valid"]),
                }

                if result["ads_dist_mae"] is not None: 
                    metrics["ads_dist_mae"] = float(result["ads_dist_mae"])
 
                for key, val in metrics.items():
                    self.log(key, val, on_step=False, on_epoch=True, batch_size=1)

                if self.gen_POS and result["is_valid"]:
                    os.makedirs(self.gen_POS, exist_ok=True)
                    ase.io.write(
                        os.path.join(self.gen_POS, f'{batch_idx}_{b}.vasp'),
                        atoms,
                        format='vasp',
                    )

                if self.rlxd_POS and self.run_mlff and result["is_valid"]:
                    os.makedirs(self.rlxd_POS, exist_ok=True)
                    ase.io.write(
                        os.path.join(self.rlxd_POS, f'{batch_idx}_{b}.vasp'),
                        atoms_rlxd,
                        format='vasp',
                    )

                if batch_idx == 2 and b == 13: 
                    print(opt_graph.ref_ads_pos[ads_batch == b])
                    print(opt_graph.ads_atomic_numbers[ads_batch == b])
                
                if batch_idx == 3 and b == 3: 
                    print(opt_graph.ref_ads_pos[ads_batch == b])
                    print(opt_graph.ads_atomic_numbers[ads_batch == b])

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
        if self.target_binding_E is not None:
            data.ads_binding_energy = torch.tensor(
                [self.target_binding_E], device=data.pos.device, dtype=torch.float32,
            )

        t = torch.linspace(
            0., 1., self.num_integration_steps,
            device=data.pos.device, dtype=torch.float32,
        )

        all_atoms = []
        all_metrics = []
        for sample_idx in range(self.num_generated_samples):
            counts = sample_prim_atom_counts(1, data.pos.device, self.prim_atoms_hist, self.num_prim_atoms)
            opt_graph = self.sample(input_graph=data, t=t, num_prim_atoms=counts, refine_final=self.refine_final)

            prim_batch = opt_graph.batch[opt_graph.tags == 1]
            ads_batch  = opt_graph.batch[opt_graph.tags == 2]

            atoms = forge_catalyst(opt_graph, 0, prim_batch, ads_batch)

            if atoms is not None:
                atoms_rlxd = relax_struc(
                    atoms=atoms,
                    run_mlff=self.run_mlff,
                    config_vasp=self.config_vasp,
                    mlff_checkpoint=self.mlff_checkpoint,
                    accelerator=self.accelerator,
                    seed=self.seed,
                )

                atoms_rlxd = atoms_rlxd if self.run_mlff else atoms
                target_ads_dists = opt_graph.ads_pairwise_dist[ads_batch == 0].cpu().numpy()
                result = check_validity(atoms_rlxd, self.covalent_radii_scale, target_ads_dists=target_ads_dists)

                metrics = {
                    "vol_ok":          float(result["vol_ok"]),
                    "cat_dist_ok":     float(result["cat_dist_ok"]),
                    "ads_dist_ok":     float(result["ads_dist_ok"]),
                    "sample_is_valid": float(result["is_valid"]),
                }
                if result["ads_dist_mae"] is not None:
                    metrics["ads_dist_mae"] = float(result["ads_dist_mae"])

                all_metrics.append(metrics)

                if self.gen_POS and result["is_valid"]:
                    os.makedirs(self.gen_POS, exist_ok=True)
                    ase.io.write(
                        os.path.join(self.gen_POS, f'{batch_idx}_{sample_idx}.vasp'),
                        atoms,
                        format='vasp',
                    )

                if self.rlxd_POS and self.run_mlff and result["is_valid"]:
                    os.makedirs(self.rlxd_POS, exist_ok=True)
                    ase.io.write(
                        os.path.join(self.rlxd_POS, f'{batch_idx}_{sample_idx}.vasp'),
                        atoms_rlxd,
                        format='vasp',
                    )

            all_atoms.append(atoms)

        if all_metrics:
            keys = all_metrics[0].keys()
            avgs = {k: np.mean([m[k] for m in all_metrics if k in m]) for k in keys}
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
