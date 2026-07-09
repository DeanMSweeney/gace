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
from gace.utils.oracle import Oracle
from gace.utils.catalyst_forge import CatalystForge
import ase.io

from typing import Optional, Literal, Union, Tuple

_AVG_BINDING_ENERGY = -1.54790583
_STD_BINDING_ENERGY = 2.27278609


class LightFlowMatch(LightningModule, Flow, Oracle):
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
            oracle_model=None,
            oracle_mode: str="off",
            variant: str="forge",
            mode: str="debug",
            track_grad_norms: bool=False,
            generated_strucs=None,
            relaxed_strucs=None,
            sanitized_strucs=None,
            config_vasp: bool=False,
            x_matching: Literal["flow", "endpoint"]="endpoint",
            schedule_type: Literal["linear", "cosine"]="linear",
            optimal_transport: bool=False,
            informed_priors: bool=True,
            num_integration_steps: int=1,
            relax_otf: bool=False,
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
            dedupe_otf: bool=False,
            fk_steering: bool=False,
            fk_mlff_checkpoint: Optional[str]="PaiNN-S2EF-OC20-All",
            fk_K: int=10,
            fk_tau: float=0.5,
            fk_start_step: int=20,
            **kwargs,
        ):

        super().__init__()
        # Save hyperparameters for logging and checkpointing (ignoring model-specific args)
        self.save_hyperparameters(ignore=["model", "oracle_model", "oracle_mode", "variant", "mode", \
            "track_grad_norms", "generated_strucs", "relaxed_strucs", "sanitized_strucs", "config_vasp", \
            "write_eval_metrics", "relax_otf", "mlff_checkpoint", \
            "fk_mlff_checkpoint", "accelerator", "seed", "num_prim_atoms"])

        # Model and training configuration
        self.model = model  # Neural network (e.g., Equiformer)
        # Energy-regression oracle (None => disabled). In "guide" mode it
        # is frozen and steers the flow model via an energy guidance loss.
        self.oracle_model = oracle_model
        self.oracle_mode = oracle_mode
        if self.oracle_model is not None and self.oracle_mode == "guide":
            # Freeze: no gradient updates. Eval mode (deterministic forward) is
            # pinned by the train() override, which Lightning triggers each epoch.
            for p in self.oracle_model.parameters():
                p.requires_grad_(False)
        elif self.oracle_model is not None and self.oracle_mode == "train" \
                and getattr(self.hparams, "oracle_gen_pretraining", False):
            # Stage A: forge trunk is frozen; train only the random energy head.
            for name, p in self.oracle_model.named_parameters():
                p.requires_grad_(name.startswith(
                    ("energy_cross_attention.", "energy_transformer.", "out_energy.")))
        self.variant = variant  # "refine" or "forge"
        self.mode = mode
        self.loss_fn = nn.MSELoss()  # Base loss function (not directly used, see criterion)

        # Monitoring and output configuration
        self.track_grad_norms = track_grad_norms
        self.generated_strucs = generated_strucs
        self.relaxed_strucs = relaxed_strucs
        self.sanitized_strucs = sanitized_strucs

        self.config_vasp = config_vasp
        self.catalyst_dist_cutoff = catalyst_dist_cutoff

        # Post processing configruation 
        self.relax_otf = relax_otf
        self.sanitize_otf = sanitize_otf
        self.dedupe_otf = dedupe_otf

        # Miscellaneous 
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

        # Separate lighter calculator for FK intermediate reward evaluation.
        # Falls back to the main calculator if fk_mlff_checkpoint is not set.
        self._fk_mlff_calculator = None
        if fk_steering:
            _fk_ckpt = fk_mlff_checkpoint or mlff_checkpoint
            if _fk_ckpt is not None:
                with Silent():
                    self._fk_mlff_calculator = OCPCalculator(
                        model_name=_fk_ckpt,
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

        # Feynman-Kac steering
        self.fk_steering   = fk_steering
        self.fk_K          = fk_K
        self.fk_tau        = fk_tau
        self.fk_start_step      = fk_start_step
        self.fk_mlff_checkpoint = fk_mlff_checkpoint

        # Mean and standard deviation of binding energy from trianing set 
        self.avg_binding_energy = -1.54790583
        self.std_binding_energy = 2.27278609

        _hist_path = os.path.join(os.path.dirname(__file__), "primitive_atom_distribution.json")
        with open(_hist_path) as f:
            hist = json.load(f)["prim_slab_num_atoms_hist"]
        self.register_buffer("prim_atoms_hist", torch.tensor(hist, dtype=torch.float32))

        # Uniqueness bookkeeping accumulated across all test/predict batches.
        # gen_slabs_memory is the running record of unique slabs shared with each
        # per-batch CatalystForge for OTF dedup; num_slabs_checked counts the
        # valid structures passed to that check. Uniqueness at epoch end is
        # len(gen_slabs_memory) / num_slabs_checked.
        self.gen_slabs_memory: list = []
        self.num_slabs_checked: int = 0
       # ==============================================================================

    def train(self, mode: bool = True):
        """Keep the frozen oracle_model in eval mode during guide-mode training.

        Lightning calls ``train()`` at each epoch start, which would otherwise flip
        the frozen oracle_model back into train mode (affecting any dropout /
        norm layers). ``requires_grad=False`` already prevents it from updating;
        this keeps its forward deterministic too.
        """
        super().train(mode)
        if self.oracle_mode == "guide" and self.oracle_model is not None:
            self.oracle_model.eval()
        return self

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

    def _run_model(self, data: Data, force_unconditional: bool = False):
        """Convert the PyG batch to dense tensors and call the flow transformer.

        This is the first half of ``forward``: it runs the model and returns the
        **dense** padded outputs together with the wrapper batches and padding
        masks. ``forward`` then sparsifies these; the energy-guidance path instead
        feeds the dense outputs straight to the oracle (no lossy round-trip).

        Returns:
            ``(pred_prim, pred_ads, x_prim, x_ads, prim_padding_mask, ads_padding_mask)``
        """
        wrapper = PyGToTensorDict()
        x_prim, x_ads, t_prim, t_ads, f_prim, f_ads = wrapper.to_tensordict(data=data)

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
            feats = self._build_target_props(f_prim=f_prim)

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

        return pred_prim, pred_ads, x_prim, x_ads, prim_padding_mask, ads_padding_mask

    def _sparsify(self, pred_prim, pred_ads, x_prim, x_ads) -> dict:
        """Unpack the dense flow outputs to the sparse prediction dict (second half
        of ``forward``)."""
        pred = {
            "pos":     x_prim.to_sparse(pred_prim[1]),   # (N_prim, 3)
            "z":       x_prim.to_sparse(pred_prim[0]),   # (N_prim, 100)
            "len":     pred_prim[2].squeeze(1),          # (B, 3)
            "ang":     pred_prim[3].squeeze(1),          # (B, 3)
            "M":       pred_prim[4],                     # (B, 3, 3)
            "ads_pos": x_ads.to_sparse(pred_ads[0]),     # (N_ads, 3)
        }
        if self.model.predict_energy:
            pred["energy"] = pred_prim[5]                # (B,)
        return pred

    def forward(self, data: Data, force_unconditional: bool = False) -> dict:
        """Run the forge model forward pass and return a flat prediction dict.

        Used both during training (via _transport_loss) and sampling (via
        Flow.sample), which calls this as `self(data)` because LightFlowMatch
        inherits from Flow.

        Args:
            data: PyG Data/Batch with all forge fields populated.
            force_unconditional: If True, drop all conditioning (cfg_drop_mask=True for
                every sample). Used for the unconditional forward pass in CFG guidance.

        Returns:
            Dict with keys: pos, z, len, ang, M, ads_pos.
        """
        pred_prim, pred_ads, x_prim, x_ads, _, _ = self._run_model(data=data, force_unconditional=force_unconditional)
        return self._sparsify(pred_prim=pred_prim, pred_ads=pred_ads, x_prim=x_prim, x_ads=x_ads)

    def criterion(self, input_graph: Data) -> torch.Tensor:
        """
        Compute the training loss for a batch of structures.

        Behaviour depends on ``oracle_mode``:
        - "train": train the oracle standalone (energy regression); the
          flow model is not run.
        - "off"/"guide": run the flow matching path; in "guide" mode also add the
          frozen-oracle energy guidance term to the flow loss.
        """
        # Mode 1: standalone oracle training — no flow loss.
        if self.oracle_mode == "train":
            energy_loss = self.energy_criterion(data=input_graph)
            return {"total_loss": energy_loss, "energy_loss": energy_loss}

        # Flow path (off / guide). path() stores supervision targets on `data`.
        data = self.build_path(input_graph=input_graph)

        if self.variant != "forge":
            raise RuntimeError(f"Unknown variant: {self.variant}")

        pred_prim, pred_ads, x_prim, x_ads, prim_pad, ads_pad = self._run_model(data=data)
        pred = self._sparsify(pred_prim=pred_prim, pred_ads=pred_ads, x_prim=x_prim, x_ads=x_ads)
        losses = self.flow_criterion(input_graph=data, pred=pred)

        # Mode 2: frozen oracle steers the flow model toward the target energy.
        if self.oracle_mode == "guide":
            energy_loss = self.energy_criterion(
                data=data,
                pred_prim=pred_prim,
                pred_ads=pred_ads,
                x_ads=x_ads,
                prim_padding_mask=prim_pad,
                ads_padding_mask=ads_pad,
            )
            losses["energy_loss"] = energy_loss
            losses["total_loss"] = losses["total_loss"] + float(self.hparams.oracle_lambda_energy) * energy_loss

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
        losses = self.criterion(data)

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
        losses = self.criterion(data)

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
        
        t = torch.linspace(0., 1, self.hparams.num_integration_steps, device=data.pos.device, dtype=data.pos.dtype)
        B = int(data.batch.max().item()) + 1
        counts = sample_prim_atom_counts(B, data.pos.device, self.prim_atoms_hist, self.num_prim_atoms)
        
        if self.target_binding_energy is not None:
            data.ads_binding_energy = torch.ones(B, device=data.pos.device, dtype=data.pos.dtype) * self.target_binding_energy
        #else: 
          #  data.ads_binding_energy = torch.normal(mean=_AVG_BINDING_ENERGY, std=_STD_BINDING_ENERGY, size=(B,), device=data.pos.device, dtype=data.pos.dtype)

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
            dedupe_otf=self.dedupe_otf,
            gen_slabs_memory=self.gen_slabs_memory, # share OTF dedup record across batches
        )
        for metrics in cf.fire():
            for key, val in metrics.items():
                if isinstance(val, float):
                    reduce_fx = "sum" if key.startswith("within_") else "mean"
                    self.log(key, val, on_step=False, on_epoch=True, batch_size=1, reduce_fx=reduce_fx)
        # Accumulate the denominator; uniqueness is reported once in on_test_epoch_end.
        self.num_slabs_checked += cf.num_slabs_checked
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

        t = torch.linspace(0., 1., self.hparams.num_integration_steps, device=data.pos.device, dtype=torch.float32)
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
            dedupe_otf=self.dedupe_otf,
            gen_slabs_memory=self.gen_slabs_memory, # share OTF dedup record across batches
        )
        all_metrics = list(cf.fire())
        all_atoms = cf._all_atoms
        # Accumulate the denominator; uniqueness is reported once in on_predict_epoch_end.
        self.num_slabs_checked += cf.num_slabs_checked

        if all_metrics:
            keys = all_metrics[0].keys()
            vals = {k: [m[k] for m in all_metrics if isinstance(m.get(k), float)] for k in keys}
            avgs = {k: (np.sum(v) if k.startswith("within_") else np.mean(v)) for k, v in vals.items() if v}
            col_w = max(len(k) for k in avgs) + 2
            sep = "-" * (col_w + 26)
            print(f"\n[predict batch={batch_idx} | {len(all_metrics)} samples]")
            print(sep)
            for k, v in avgs.items():
                print(f"{k:>{col_w}} | {v:.4f}")
            print(sep)

        return all_atoms

    # ==================== Uniqueness bookkeeping ====================

    def _reset_uniqueness_memory(self):
        self.gen_slabs_memory.clear()
        self.num_slabs_checked = 0

    def _run_uniqueness(self) -> dict | None:
        """Uniqueness over every valid structure generated this run.

        gen_slabs_memory holds exactly the unique slabs (OTF dedup appends only
        non-duplicates), so the ratio is len(unique) / (# valid checked).
        """
        if not self.dedupe_otf or self.num_slabs_checked == 0:
            return None
        n_unique = len(self.gen_slabs_memory)
        return {f"uniqueness ({self.num_slabs_checked})": n_unique / self.num_slabs_checked}

    def on_test_epoch_start(self):
        self._reset_uniqueness_memory()

    def on_test_epoch_end(self):
        uniqueness = self._run_uniqueness()
        if uniqueness is not None:
            for key, val in uniqueness.items():
                self.log(key, val, on_step=False, on_epoch=True)

    def on_predict_epoch_start(self):
        self._reset_uniqueness_memory()

    def on_predict_epoch_end(self):
        uniqueness = self._run_uniqueness()
        if uniqueness is not None:
            for key, val in uniqueness.items():
                print(f"{key} | {val:.4f}")

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
        # Optimise the oracle alone when training it standalone; otherwise
        # the flow model (the oracle is frozen in "guide" mode and absent
        # when off, so self.model.parameters() is the right set in both cases).
        if self.oracle_mode == "train":
            # Only trainable params (Stage A freezes the forge trunk).
            params = [p for p in self.oracle_model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()

        # AdamW optimizer: Adam with improved weight decay handling
        optimizer = optim.AdamW(
            params,
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