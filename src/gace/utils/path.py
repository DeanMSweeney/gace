import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from gace.utils.invariance import pbc_aware_delta, batched_ot_align, global_ot_align
from gace.utils.utils import adsorbate_neighborhood_mask, center_and_random_so3_rotation
from gace.utils.builder import lattice_params_to_matrix
from gace.utils.euler import euler_step, ctmc_step
from gace.utils.relaxation import relax_w_mlff
from gace.utils.catalyst_forge import CatalystForge

class Flow():
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms
    """
    def flow_criterion(
        self,
        input_graph,
        pred: dict,
    ):
        """
        Loss for the forge variant: positions + atom types + lattice + adsorbate.

        Components
        ----------
        x_loss      : MAE on catalyst fractional coordinates (endpoint)
        z_loss      : cross-entropy on catalyst atomic numbers (adsorbate types are conditioned)
        ads_pos_loss: MAE on adsorbate absolute Cartesian positions
        len_loss    : MAE on primitive-cell lengths
        ang_loss    : MAE on primitive-cell angles
        M_loss      : MAE on supercell transformation matrix

        total = x + λ_z·z + λ_ads·(ads_com + ads_relp) + λ_lat·(len + ang) + λ_M·M
        """
        data = input_graph

        prim_batch = data.batch[data.tags == 1]  # (N_prim,)
        ads_batch  = data.batch[data.tags == 2]  # (N_ads,)

        # ── Position loss (endpoint) w/ minimum image convention ─────────────────────────────────────────
        x_pred = pred["pos"].float()
        x_target = data.pos_target.float()
        x_diff = x_pred - x_target
        x_diff = x_diff - torch.round(x_diff) if self.hparams.informed_priors else x_diff # apply MIC when xdiff is b/t [0,1]
        x_loss_per_graph = scatter(x_diff.abs().mean(dim=1), prim_batch, dim=0, reduce="mean")  # (B,)
        x_loss = (x_loss_per_graph * self.time_weight).mean()
        # ─────────────────────────────────────────────────────────────────────────────────────────────────

        # ── Atom-type loss (catalyst only — adsorbate types are conditioned) ─────────────────────────────
        z_pred   = pred["z"].float()
        z_target = data.atomic_numbers_target.long().clone()         # (N_prim,)
        z_target[data.atomic_numbers != self.model.atom_dim - 1] = -100      # ignore already-unmasked atoms
        z_loss_atoms = F.cross_entropy(z_pred, z_target, ignore_index=-100, reduction='none')  # (N_prim,)
        valid = (z_target != -100).float()
        z_loss_per_graph = (
            scatter(z_loss_atoms * valid, prim_batch, dim=0, reduce="sum")
            / scatter(valid, prim_batch, dim=0, reduce="sum").clamp(min=1)
        )  # (B,)
        z_loss = (z_loss_per_graph * self.time_weight).mean()
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # ── Lattice losses (endpoint) ───────────────────────────────────────────────────────────────────
        len_loss = ((pred["len"] - data.lengths_scaled_target).abs().mean(dim=1) * self.time_weight).mean()
        ang_loss = ((pred["ang"] - data.angles_radians_target).abs().mean(dim=1) * self.time_weight).mean()
        M_loss   = ((pred["M"]   - data.transformation_matrix_target).abs().mean(dim=(1, 2)) * self.time_weight).mean()
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # ── Adsorbate losses (endpoint) ─────────────────────────────────────────────────────────────────
        ads_loss_per_graph = scatter(
            (pred["ads_pos"] - data.ads_pos_target).abs().mean(dim=1), ads_batch, dim=0, reduce="mean"
        )  # (B,)
        ads_pos_loss = (ads_loss_per_graph * self.time_weight).mean()
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        lambda_x   = float(self.hparams.lambda_x)
        lambda_z   = float(self.hparams.lambda_z)
        lambda_ads = float(self.hparams.lambda_ads)
        lambda_lat = float(self.hparams.lambda_lat)
        lambda_M   = float(self.hparams.lambda_M)

        total_loss = (
              lambda_x   * x_loss
            + lambda_z   * z_loss
            + lambda_ads * ads_pos_loss
            + lambda_lat * (len_loss + ang_loss)
            + lambda_M   * M_loss
        )

        return {
            "total_loss":   total_loss,
            "x_loss":       x_loss,
            "z_loss":       z_loss,
            "ads_pos_loss": ads_pos_loss,
            "len_loss":     len_loss,
            "ang_loss":     ang_loss,
            "M_loss":       M_loss,
        }

    def build_path(
        self,
        input_graph: Data,
        time_dist: Literal["uniform", "beta"] = "beta",
        beta_t: float = 1.8,
    ) -> Data:
        """
        Generate interpolated training data along a path from initial to relaxed positions.

        For "forge" variant: interpolates from noise/initial positions to relaxed positions

        Returns:
            data: Modified data object with interpolated inputs/targets at time t
        """

        with torch.no_grad():
            data = input_graph.clone()

            # define time 
            if time_dist == "uniform":
                t = torch.rand(len(torch.unique(data.batch)), 1, device=data.pos.device, dtype=torch.float32) # B , 1
            elif time_dist == "beta":
                beta = torch.distributions.Beta(beta_t, 1.0)
                t = beta.sample((len(torch.unique(data.batch)), 1)).to(data.pos.device) # B , 1
            data.t = t 

            # ── Lattice and supercell (CatFlow data only) ────────────────────
            self.primitive_mask = (data.tags == 1)
            self.adsorbate_mask = (data.tags == 2)
        
            data = self._build_lattice_path(data, t=t) # lattice 
            data = self._build_transformation_matrix_path(data, t=t) # transformation matrix 
            data = self._build_ads_path(data, t=t) # adsorbate position 
            data = self._build_positions_path(data, t=t) # positions
            data = self._build_atom_type_path(data, t=t) # atom types 

            return data


    def _interpolant_schedule(
        self,
        t: Tensor,
        schedule_type: str,
        v: float = 1,
        x_matching: str = "flow",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute interpolation schedule parameters.

        Args:
            t: Interpolation time tensor
            schedule_type: "linear" or "cosine" schedule type
            v: Exponent for cosine schedule (controls interpolation curvature)

        Returns:
            alpha_t: Interpolation parameter
            alpha_t_grad: Gradient d(alpha_t)/dt for velocity rescaling
            alpha_t_factor: Time-dependent factor for loss weighting
        """
        if schedule_type == "linear":
            # Linear interpolation: alpha_t = t
            alpha_t = t
            alpha_t_grad = torch.tensor(1.0, device=t.device, dtype=torch.float32)
        
        elif schedule_type == "cosine":
            # Cosine interpolation: slower at start/end, faster in middle
            alpha_t = 1 - (torch.cos(t ** v * torch.pi / 2)) ** 2
            alpha_t_grad = torch.pi * v * t ** (v - 1) * torch.cos(t ** v * torch.pi / 2) * torch.sin(t ** v * torch.pi / 2)
        
        else:
            raise ValueError("Invalid interpolant schedule")

        # scale only for endpoint
        alpha_t_factor = torch.clamp(alpha_t_grad / (1 - alpha_t),  min=None, max=50) if x_matching == "endpoint" \
            else torch.tensor(1.0, device=t.device, dtype=torch.float32)

        return alpha_t, alpha_t_grad, alpha_t_factor
            
    # ─────────────────────────────────────────────────────────────────────────
    # Modular flow helpers — each takes a Data object and returns it updated.
    # ─────────────────────────────────────────────────────────────────────────

    def _build_lattice_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1) -> Data:
        """
        Flow-match the primitive cell matrix (3×3 Cartesian, Å).

        x0  ~ N(0, I)  scaled to match the RMS magnitude of the true cell.
        x1  = data.prim_cell  (true primitive cell, shape (3,3) or (B,3,3))
        xt  = x0 + alpha_t * (x1 - x0)   (schedule-interpolated)
        u_L = alpha_t_grad * (x1 - x0)   (rescaled target velocity)

        The noisy cell `xt` is stored back as `data.prim_cell` so that
        downstream graph construction uses it at the current flow time.

        """

        # sample edge lenghts and interaxial angles from true and noisy distribution
        # pyg_converter stores lengths_scaled already normalised by N^(1/3) → O(1) values
        len1 = data.lengths_scaled.float()  # (B, 3)
        ang1 = data.angles_radians.float()  # (B, 3)

        if self.hparams.informed_priors:
            ang0 = torch.empty_like(ang1).uniform_(torch.pi / 3, 2 * torch.pi / 3)  # Uniform(60°, 120°)
            len0 = torch.empty_like(len1).uniform_(1.5, 5.0)  # Uniform(1.5, 5.0) — always positive, covers typical scaled lengths
        else:
            ang0 = torch.randn_like(ang1)  # N(0, 1)
            len0 = torch.randn_like(len1)  # N(0, 1)

        t_len_per_graph = t # B, 1
        t_len = t_len_per_graph

        alpha_t_len, alpha_t_len_grad, alpha_t_len_loss_coeff = self._interpolant_schedule(
            t=t_len,
            schedule_type=self.hparams.schedule_type,
            x_matching=self.hparams.x_matching,
            v=cosine_schd_freq, 
        )

        t_ang_per_graph = t # B, 1
        t_ang = t_ang_per_graph

        alpha_t_ang, alpha_t_ang_grad, alpha_t_ang_loss_coeff = self._interpolant_schedule(
            t=t_ang,
            schedule_type=self.hparams.schedule_type,
            x_matching=self.hparams.x_matching,
            v=cosine_schd_freq, 
        )

        u_len = (len1 - len0)
        u_ang = (ang1 - ang0)

        len_t = len0 + alpha_t_len * u_len
        ang_t = ang0 + alpha_t_ang * u_ang

        data.lengths_scaled = len_t
        data.angles_radians = ang_t
        data.lengths_scaled_target = len1
        data.angles_radians_target = ang1


        # store velocity fields 
        if self.hparams.x_matching == "flow":
            data.u_len = alpha_t_len_grad * u_len
            data.u_ang =alpha_t_ang_grad * u_ang


        return data

    def _build_transformation_matrix_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1)-> Data:
        """
        Flow-match the supercell matrix M  (3×3, integer-valued targets).

        x0  ~ I + 0.1·N(0,1)  (prior near-identity; supercells are usually close to I)
        x1  = data.supercell_matrix  (true integer supercell matrix)
        xt  = x0 + alpha_t * (x1 - x0)   (schedule-interpolated)
        u_M = alpha_t_grad * (x1 - x0)   (rescaled target velocity)
        """
        device = data.transformation_matrix.device
        dtype  = torch.float32

        # pyg_converter stores as (1, 3, 3) per graph → batches to (B, 3, 3)
        M1 = data.transformation_matrix.float()  # (B, 3, 3)
        B = M1.size(0)

        # torch.eye gives (3, 3); unsqueeze + expand to (B, 3, 3)
        identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        M0 = identity + 0.1 * torch.randn_like(M1)

        t_M_per_graph = t # B, 1
        t_M = t_M_per_graph

        alpha_t_M, alpha_t_M_grad, alpha_t_M_loss_coeff = self._interpolant_schedule(
            t=t_M,
            schedule_type=self.hparams.schedule_type,
            x_matching=self.hparams.x_matching,
            v=cosine_schd_freq, 
        )

        self.time_weight = alpha_t_M_loss_coeff.squeeze(-1)  # (B,) or scalar

        u_M = M1 - M0
        M_t = M0 + alpha_t_M.unsqueeze(-1) * u_M

        data.transformation_matrix = M_t
        data.transformation_matrix_target = M1

        # Build supercell lattice from primitive cell and transformation matrix
        # prim_cell_t: (B, 3, 3) from lattice parameters
        # M_t: (B, 3, 3) transformation matrix
        # supercell = M_t @ prim_cell_t

        # First, build primitive cell from lattice parameters
        # lengths_scaled is normalised by N^(1/3) — denormalise before geometry construction
        n_scale = data.prim_natoms.float().unsqueeze(1) ** (1 / 3)  # (B, 1)
        C_t = lattice_params_to_matrix(
            lengths=data.lengths_scaled * n_scale,  # (B, 3) back to Å
            angles=data.angles_radians,              # (B, 3)
        )  # (B, 3, 3)

        # Then apply transformation matrix to get supercell lattice
        # Supercell lattice: L_super = M @ L_prim
        data.supercell_lattice = torch.bmm(M_t, C_t)  # (B, 3, 3)
        data.cell = C_t  # Store primitive cell as well 

        if self.hparams.x_matching == "flow":
            data.u_M = u_M * alpha_t_M_grad.unsqueeze(-1)

        return data

    def _build_positions_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1) -> Data:
        """
        Flow-match atomic positions for both 'place' and 'forge' variants.

        place variant:
            x0 = data.pos  (initial DFT structure)
            x1 = data.pos_relaxed
            Supervises only the adsorbate neighbourhood (ads_cutoff).

        forge variant:
            x0 = Gaussian noise for catalyst atoms, data.pos for adsorbate.
            Hungarian alignment applied to catalyst atoms.
            Supervises all non-frozen atoms.

        Stores on data:
            data.pos        ← noisy x_t  (model input)
            data.pos_init   ← x0
            data.t          ← per-atom interpolation time
            data.u_x        ← target velocity (rescaled by schedule gradient)
        """
        if self.variant == "forge":
            x1 = data.pos.float() # combined primtive cell 

            if self.hparams.informed_priors:
                x0 = torch.rand_like(x1, dtype=torch.float32)  # Uniform[0, 1) — fractional coords live on torus
            else:
                x0 = torch.randn_like(x1, dtype=torch.float32)  # N(0, 1)

            if self.hparams.optimal_transport:
                x0 = batched_ot_align(source=x0, target=x1, batch=data.batch[self.primitive_mask], pbc_aware=True)

            u_x = x1 - x0
            u_x = u_x - torch.round(u_x) if self.hparams.informed_priors else u_x

            # Sample per-graph interpolation time
            t_x_per_graph = t # B, 1 

            # broadcast to per atom
            t_x = t_x_per_graph[data.batch[self.primitive_mask]] # Nprim, 1
        
        else:
            raise ValueError("Unrecognized variant selected")

        alpha_t_x, alpha_t_x_grad, alpha_t_x_loss_coeff = self._interpolant_schedule(
            t=t_x,
            schedule_type=self.hparams.schedule_type,
            x_matching=self.hparams.x_matching,
            v=cosine_schd_freq, 
        )

        # interpolation 
        xt  = x0 + alpha_t_x * u_x       # rescaled velocity

        data.pos   = xt   
        data.pos_target = x1

        if self.hparams.x_matching == "flow":
            data.u_x = u_x * alpha_t_x_grad

        return data

    def _build_ads_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1) -> Data:
        """
        Flow-match adsorbate absolute positions (forge variant only).

        Path, loss, and integration all run in z-scored space (using
        self.ads_pos_mean / self.ads_pos_std as per-axis stats). Denormalization
        back to raw Å happens only at the very end of _sample_forge.

        Stores on data:
            data.ads_pos        ← interpolated positions x_t (normalized, model input)
            data.ads_pos_target ← true positions x_1 (normalized, supervision target)
        """
        ads_batch = data.batch[self.adsorbate_mask]  # (N_ads,)

        # Read absolute Cartesian target; optionally z-score with dataset stats
        ads_pos1 = data.ads_pos.float()                                  # (N_ads, 3) raw Å
        ads_pos0 = torch.randn_like(ads_pos1)                            # N(0,1) prior

        #if self.hparams.optimal_transport:
        #ads_pos0 = batched_ot_align(source=ads_pos0, target=ads_pos1, batch=ads_batch, pbc_aware=False)

        t_ads = t[ads_batch]  # (N_ads, 1)
        alpha_t_ads, alpha_t_ads_grad, _ = self._interpolant_schedule(
            t=t_ads,
            schedule_type=self.hparams.schedule_type,
            x_matching=self.hparams.x_matching,
            v=cosine_schd_freq,
        )

        u_ads = ads_pos1 - ads_pos0
        ads_pos_t = ads_pos0 + alpha_t_ads * u_ads

        data.ads_pos = ads_pos_t
        data.ads_pos_target = ads_pos1

        if self.hparams.x_matching == "flow":
            data.u_ads_pos = alpha_t_ads_grad * u_ads

        # Center ref_ads_pos and apply random SO(3) rotation each training step.
        # Stored as ads_ref_centered_pos — conditioning signal, distinct from ads_pos.
        B = t.shape[0]
        data.ads_ref_centered_pos = center_and_so3_rotation(
            data.ref_ads_pos.float(), ads_batch, B
        )  # (N_ads, 3)

        return data

    def _build_atom_type_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1) -> Data:
        """
        CTMC mask-token corruption for catalyst atom types (forge variant only).

        Real atom types are indices 0–98 (1-indexed elements 1–99).
        Mask token is self.model.atom_dim - 1 = 99.

        Each atom is independently masked with probability 1 - alpha_t(t),
        where alpha_t is computed from the interpolant schedule per graph then
        broadcast to per-atom. Masked atoms receive index 99; unmasked atoms
        keep their true type. Loss is applied only to currently masked atoms.

        Adsorbate atom types are conditioned (fixed to their true value).
        """

        device = data.pos.device

        z1 = (data.atomic_numbers.long() - 1)  # (N_prim,) 0-indexed true types (0–98)

        # Sample per-graph interpolation time
        t_z_per_graph = t
        t_z = t_z_per_graph[data.batch[self.primitive_mask]] # Nprim, 1

        alpha_t_z, alpha_t_z_grad, alpha_t_z_loss_coeff = self._interpolant_schedule(
            t=t_z,
            schedule_type=self.hparams.schedule_type,
            x_matching="endpoint",
            v=cosine_schd_freq,
        )

        # Per-atom independent masking against the scheduled threshold
        mask_flag = torch.rand(z1.size(0), device=device) < (1 - alpha_t_z.squeeze(1))  # (N_prim,)
        zt = z1.clone()
        zt[mask_flag] = self.model.atom_dim - 1          # masked atoms get index 99

        data.atomic_numbers        = zt          # (N_prim,) long — model input
        data.atomic_numbers_target = z1          # (N_prim,) long — loss target

        # Condition on adsorbate atom types
        data.ads_atomic_numbers = data.ads_atomic_numbers.long() - 1  # shift 1-indexed → 0-indexed

        return data

    def sample(
            self,
            input_graph: Data,
            t: torch.Tensor,  # Time points, e.g. linspace(0, 1, n_steps)
            num_prim_atoms: Optional[Union[int, torch.Tensor]] = None,
    ) -> Data:
        """Generate structures for the forge variant via a custom Euler loop.

        All six modalities are integrated simultaneously:
        - Continuous (pos, lengths, angles, M, ads_com, ads_rel_pos): deterministic
          endpoint-matching Euler steps.
        - Discrete (atom types): stochastic CTMC steps using the model's predicted
          endpoint distribution, following zatom's DiscreteInterpolant.step().

        Args:
            input_graph: PyG Data/Batch. Adsorbate conditioning fields
                (ads_atomic_numbers, ads_ref_centered_pos, ads_binding_energy) are
                used as-is; all generated modalities are re-initialised from prior.
            t: 1-D tensor of monotonically increasing time points in [0, 1].
                The loop takes len(t)-1 Euler steps.

        Returns:
            data with *_generated fields set for each modality.
        """

        build_from_truth = False
        if build_from_truth: 
            num_prim_atoms = input_graph.prim_natoms

        data = input_graph.clone()
        device = data.pos.device

        self.primitive_mask = (data.tags == 1)
        self.adsorbate_mask = (data.tags == 2)
        B = int(data.batch.max().item()) + 1

        # Center ref_ads_pos for conditioning (no rotation at inference)
        ads_batch = data.batch[self.adsorbate_mask]
        ref_ads_pos = data.ref_ads_pos.float()
        com = scatter(ref_ads_pos, ads_batch, dim=0, dim_size=B, reduce="mean")  # (B, 3)
        data.ads_ref_centered_pos = ref_ads_pos - com[ads_batch]                 # (N_ads, 3)

        # Shift ads conditioning to 0-indexed (matches _build_atom_type_path)
        data.ads_atomic_numbers = data.ads_atomic_numbers.long() - 1
        N_ads = int(self.adsorbate_mask.sum().item())

        if not build_from_truth: # TOGGLED ON ONLY FOR TESTING THE BUILD PROCESS, NORMALLY OFF 
            if num_prim_atoms is not None:
                total_prim = int(num_prim_atoms.sum().item())
                ads_mask = (data.tags == 2)
                new_prim_batch = torch.repeat_interleave(torch.arange(B, device=device), num_prim_atoms)
                new_prim_tags = torch.ones(total_prim, dtype=data.tags.dtype, device=device)
                data.batch = torch.cat([new_prim_batch, data.batch[ads_mask]])
                data.tags  = torch.cat([new_prim_tags,  data.tags[ads_mask]])
                data.pos   = torch.zeros(total_prim, 3, device=device)
                self.primitive_mask = (data.tags == 1)
                self.adsorbate_mask = (data.tags == 2)

            # ── Initialise all generated modalities from prior (t = 0) ───────────
            # Continuous
            if self.hparams.informed_priors:
                data.pos = torch.rand_like(data.pos)                            # Uniform[0,1) — fractional coords on torus
                data.angles_radians = torch.empty(B, 3, device=device).uniform_(torch.pi / 3, 2 * torch.pi / 3)  # Uniform(60°, 120°)
                data.lengths_scaled = torch.empty(B, 3, device=device).uniform_(1.5, 5.0)
            else:
                data.pos            = torch.randn_like(data.pos)                # N(0, 1)
                data.angles_radians = torch.randn(B, 3, device=device)         # N(0, 1)
                data.lengths_scaled = torch.randn(B, 3, device=device)  # N(0, 1) in normalised space
        
            data.ads_pos = torch.randn(N_ads, 3, device=device)             # N(0, 1)
            identity = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
            data.transformation_matrix = (identity + 0.1 * torch.randn(B, 3, 3, device=device)).float()   # (B, 3, 3)

            # Discrete: all atoms start as mask token
            N_prim = int(self.primitive_mask.sum().item())
            data.atomic_numbers = torch.full((N_prim,), self.model.atom_dim - 1, dtype=torch.long, device=device)

        # N^(1/3) scale factor for length denormalisation — lengths_scaled is in normalised space
        n_scale = (num_prim_atoms.float() ** (1 / 3)).unsqueeze(1)  # (B, 1)

        # ── Euler integration loop ────────────────────────────────────────────
        if not build_from_truth:
            # Pre-compute FK harmonic-sum schedule if steering is enabled.
            # L = number of resampling events; H_L = L-th harmonic number.
            # log_w accumulates per-particle path costs across all events.
            fk_log_w = fk_ell = fk_L = fk_H_L = None
            if self.fk_steering:
                assert data.ads_binding_energy.unique().numel() == 1, (
                    "FK steering requires all particles to share a single target "
                    "binding energy. Set target_binding_energy in sampler_cfg."
                )
                if self.sde_noise_scale == 0.0:
                    import warnings
                    warnings.warn(
                        "fk_steering=True but sde_noise_scale=0.0: resampled clones will not "
                        "diverge on continuous modalities. Set sde_noise_scale=0.05.",
                        UserWarning,
                    )
                fk_event_steps = [
                    i for i in range(len(t) - 1)
                    if (i + 1) >= self.fk_start_step
                    and (i + 1) % self.fk_K == 0
                    and i < len(t) - 2
                ]
                fk_L   = len(fk_event_steps)
                fk_H_L = sum(1.0 / k for k in range(1, fk_L + 1)) if fk_L > 0 else 1.0
                fk_log_w = torch.zeros(B, device=device)
                fk_ell   = 0

            for i in range(len(t) - 1):
                t_curr = t[i].item()
                dt     = (t[i + 1] - t[i]).item()

                # Rebuild derived lattice fields from current state
                # lengths_scaled is in normalised space — denormalise before geometry construction
                C_t = lattice_params_to_matrix(
                    lengths=data.lengths_scaled * n_scale,  # back to Å
                    angles=data.angles_radians,
                )  # (B, 3, 3)
                data.cell             = C_t
                data.supercell_lattice = torch.bmm(data.transformation_matrix, C_t)

                # Per-graph time for the model
                data.t = torch.full((B, 1), t_curr, device=device, dtype=torch.float32)

                # Model forward — optionally apply classifier-free guidance
                with torch.no_grad():
                    pred_cond = self(data)

                    w1 = self.cfg_w_continuous
                    w2 = self.cfg_w_discrete
                    if w1 != 1.0 or w2 != 1.0:
                        pred_uncond = self(data, force_unconditional=True)

                        pred = {
                            key: (1 - w1) * pred_uncond[key] + w1 * pred_cond[key]
                            for key in ("pos", "len", "ang", "M", "ads_pos")
                        }

                        log_p_cond   = F.log_softmax(pred_cond["z"],   dim=-1)
                        log_p_uncond = F.log_softmax(pred_uncond["z"], dim=-1)
                        pred["z"]    = (1 - w2) * log_p_uncond + w2 * log_p_cond
                    else:
                        pred = pred_cond

                # Update continuous modalities
                noise_scale = self.sde_noise_scale
                data.pos               = euler_step(data.pos,               pred["pos"],     t_curr, dt, pbc_aware=True,  noise_scale=noise_scale)
                data.lengths_scaled    = euler_step(data.lengths_scaled,    pred["len"],     t_curr, dt, pbc_aware=False, noise_scale=noise_scale)
                data.angles_radians    = euler_step(data.angles_radians,    pred["ang"],     t_curr, dt, pbc_aware=False, noise_scale=noise_scale)
                data.transformation_matrix = euler_step(data.transformation_matrix, pred["M"], t_curr, dt, pbc_aware=False, noise_scale=noise_scale)
                data.ads_pos = euler_step(data.ads_pos, pred["ads_pos"], t_curr, dt, pbc_aware=False, noise_scale=noise_scale)

                # Update discrete atom types (CTMC)
                data.atomic_numbers = ctmc_step(data.atomic_numbers, pred["z"], t_curr, dt)

                # ── FK steering hook ─────────────────────────────────────────
                if (
                    self.fk_steering
                    and (i + 1) >= self.fk_start_step
                    and (i + 1) % self.fk_K == 0
                    and i < len(t) - 2
                ):
                    n_prim = int(self.primitive_mask.sum().item()) // B
                    n_ads  = N_ads // B
                    rewards = self._fk_evaluate(
                        data=data, pred=pred, N=B,
                        n_prim=n_prim, n_ads=n_ads, n_scale=n_scale,
                    )
                    # Harmonic-sum weight accumulation: later events get a larger
                    # share of the total inverse-temperature budget β = 1/fk_tau.
                    beta  = 1.0 / self.fk_tau
                    costs = -rewards   # cost ≥ 0; -inf particles get large cost
                    costs[costs.isinf()] = costs[~costs.isinf()].max() + 1.0
                    fk_log_w += -beta * costs / ((fk_L - fk_ell) * fk_H_L)

                    data, fk_log_w = self._fk_resample(
                        data=data, log_w=fk_log_w, N=B,
                        n_prim=n_prim, n_ads=n_ads,
                    )
                    fk_ell += 1

        # ── Store generated outputs ───────────────────────────────────────────
        data.pos_generated             = data.pos
        data.atomic_numbers_generated  = data.atomic_numbers + 1 if not build_from_truth else data.atomic_numbers  # 1-indexed (1–99)
        data.lengths_generated         = data.lengths_scaled * n_scale  # denormalise back to Å
        data.angles_generated          = data.angles_radians
        data.transformation_matrix     = data.transformation_matrix.round().long()
        data.M_generated               = data.transformation_matrix
        data.ads_pos_generated         = data.ads_pos


        return data

    # ── Feynman-Kac helpers ──────────────────────────────────────────────────

    def _fk_evaluate(self, data, pred, N: int, n_prim: int, n_ads: int, n_scale) -> torch.Tensor:
        """Decode x1_pred for each particle, relax, and return volcano rewards.

        Writes pred into *_generated fields on data (temporary), initialises
        CatalystForge with sanitize_otf=False and relax_otf=True, then calls
        build_from_factorized_repr() + relax_w_mlff() per particle.

        Args:
            data:    current PyG Batch (N particles)
            pred:    model endpoint predictions dict
            N:       number of particles (batch size)
            n_prim:  primitive atoms per particle (uniform)
            n_ads:   adsorbate atoms per particle (uniform)
            n_scale: (B, 1) length de-normalisation scale factor

        Returns:
            rewards: (N,) float tensor, reward = -|E_ads - target_E|
        """
        rewards    = torch.full((N,), float("-inf"), dtype=torch.float32, device=data.pos.device)
        prim_batch = data.batch[data.tags == 1]
        ads_batch  = data.batch[data.tags == 2]

        # Temporarily write x1_pred into *_generated fields
        data.pos_generated            = pred["pos"]
        data.atomic_numbers_generated = pred["z"].argmax(dim=-1) + 1   # 1-indexed
        data.lengths_generated        = pred["len"] * n_scale           # denormalise to Å
        data.angles_generated         = pred["ang"]
        data.M_generated              = pred["M"].round().long()
        data.ads_pos_generated        = pred["ads_pos"]

        forge = CatalystForge(
            opt_graph=data,
            binding_energies=[data.ads_binding_energy[b].item() for b in range(N)],
            batch_idx=0,
            calculator=self.fk_mlff_checkpoint,
            sanitize_otf=False,
            relax_otf=True,
            catalyst_dist_cutoff=self.catalyst_dist_cutoff,
            generated_strucs=None,
            sanitized_strucs=None,
            relaxed_strucs=None,
        )

        for b in range(N):
            atoms = forge.build_from_factorized_repr(
                b=b, prim_batch=prim_batch, ads_batch=ads_batch,
            )
            if atoms is None:
                continue

            result, _ = relax_w_mlff(atoms=atoms, calculator=self._fk_mlff_calculator, steps=100, fmax=0.50)
            
            E_ads    = result.get_potential_energy()
            target_E = data.ads_binding_energy[b].item()
            rewards[b] = -abs(E_ads - target_E)   # volcano: peaked at conditioning label

        return rewards

    def _fk_resample(self, data, log_w: torch.Tensor, N: int, n_prim: int, n_ads: int):
        """Stratified resampling of the N-particle batch using accumulated log-weights.

        Particles are selected proportionally to exp(log_w). Stratified resampling
        gives lower variance than multinomial while keeping the same expectation.
        Selected particles carry their accumulated log-weight history forward so
        subsequent events build on path cost rather than resetting each time.

        Args:
            data:  current PyG Batch (N particles, uniform atom counts)
            log_w: (N,) accumulated log-weight tensor from harmonic-sum scheduler
            N:     number of particles
            n_prim: primitive atoms per particle
            n_ads:  adsorbate atoms per particle

        Returns:
            (data, log_w) with particle tensors and weights reordered by selection
        """
        # Convert log-weights to normalised probabilities (shift by max for stability)\
        log_w_shifted = log_w - log_w.max()
        probs = torch.exp(log_w_shifted)
        probs = probs / probs.sum()

        # Stratified resampling: single uniform offset shared across N evenly-spaced
        # positions on the CDF — lower variance than independent multinomial draws.
        cdf = torch.cumsum(probs, dim=0)
        u   = torch.rand(1, device=probs.device) / N
        positions = u + torch.arange(N, device=probs.device, dtype=torch.float32) / N
        selected  = torch.searchsorted(cdf.contiguous(), positions.contiguous()).clamp(0, N - 1)

        # Per-graph tensors
        data.lengths_scaled        = data.lengths_scaled[selected]
        data.angles_radians        = data.angles_radians[selected]
        data.transformation_matrix = data.transformation_matrix[selected]

        # Per-atom tensors: reshape to (N, n_atoms, ...), index, reshape back
        data.pos            = data.pos.view(N, n_prim, 3)[selected].view(N * n_prim, 3)
        data.atomic_numbers = data.atomic_numbers.view(N, n_prim)[selected].view(N * n_prim)
        data.ads_pos        = data.ads_pos.view(N, n_ads, 3)[selected].view(N * n_ads, 3)

        # Carry selected particles' accumulated log-weights forward
        log_w_new = log_w[selected]

        return data, log_w_new
