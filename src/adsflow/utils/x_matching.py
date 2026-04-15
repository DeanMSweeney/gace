import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from adsflow.utils.invariance import pbc_aware_delta, batched_ot_align
from adsflow.utils.utils import adsorbate_neighborhood_mask
from adsflow.utils.builder import lattice_params_to_matrix
from adsflow.utils.forge_sampler import euler_step, ctmc_step

NUM_CLASSES = 90  # number of atom-type classes (atomic numbers 1–90, shifted to 0-indexed)

class XMatching():
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms
    """

    def criterion(
        self,
        input_graph,
        pred: Union[Tensor, dict],  # N, 3 positions or dict with {"u_x", "z_logits", ...}
    ):
        """Dispatch to the variant-specific loss."""
        if self.variant == "place":
            return self._criterion_place(input_graph, pred)
        elif self.variant == "forge":
            return self._criterion_forge(input_graph, pred)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _criterion_place(
        self,
        input_graph,
        pred: Union[Tensor, dict],
    ):
        """
        Loss for the relax variant: atomic positions only.

        - Supervises only the adsorbate neighbourhood (controlled by ads_cutoff).
        - Adsorbate atoms are upweighted by tao relative to catalyst neighbours.
        - Supports both flow (velocity) and endpoint (position) matching modes.
        """
        data = input_graph
        pred_x = pred.float()

        if self.x_matching == "endpoint":
            target_x = data.pos_relaxed.float()
        else:
            if not hasattr(data, "u_x"):
                raise ValueError("Flow criterion requires `data.u_x` to be set in path().")
            target_x = data.u_x.float()

        if self.x_matching == "endpoint":
            diff = pbc_aware_delta(
                cell=data.cell.float(),
                delta=(pred_x - target_x),
                batch=data.batch,
            )
        elif self.x_matching == "flow":
            diff = pred_x - target_x
        else:
            raise ValueError("Invalid x_matching routine")

        with torch.no_grad():
            alpha_t_x_factor = getattr(self, "alpha_t_x_factor", 1.0)

            # Neighbourhood mask: adsorbate + optional cutoff shell of catalyst atoms
            if hasattr(data, "ads_neighborhood_mask"):
                mask = data.ads_neighborhood_mask.float().unsqueeze(1)
            else:
                mask = (data.fixed == 0).float().unsqueeze(1)

            # Upweight adsorbate atoms by tao
            tao = self.hparams.tao
            adsorbates = (data.tags == 2).float().unsqueeze(1)  # N, 1
            mask_factor = mask * (1 + adsorbates * (tao - 1))

        L_per_atom = diff.abs().mean(dim=1, keepdim=True)           # N, 1
        modified_L_per_atom = L_per_atom * mask_factor * alpha_t_x_factor
        x_loss = modified_L_per_atom.sum() / mask.sum().clamp_min(1e-8)

        with torch.no_grad():
            x_max_L = modified_L_per_atom.max()

        return x_loss, {"x_loss": x_loss}

    def _criterion_forge(
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
        ads_com_loss: MAE on adsorbate centre-of-mass position
        ads_relp_loss: MAE on adsorbate relative positions
        len_loss    : MAE on primitive-cell lengths
        ang_loss    : MAE on primitive-cell angles
        M_loss      : MAE on supercell transformation matrix

        total = x + λ_z·z + λ_ads·(ads_com + ads_relp) + λ_lat·(len + ang) + λ_M·M
        """
        data = input_graph

        # ── Position loss (endpoint) w/ minimum image convention ─────────────────────────────────────────
        x_pred = pred["pos"].float()
        x_target = data.pos_target.float() # fractiona coordiantes of primitive cell
        x_diff = x_pred - x_target
        n0 = torch.round(x_diff)
        x_diff = x_diff - n0
        x_loss = x_diff.abs().mean()  # scalar
        with torch.no_grad():
            x_max_L = x_diff.abs().mean(dim=1).max()  # per-atom max for monitoring
        # ─────────────────────────────────────────────────────────────────────────────────────────────────

        # ── Atom-type loss (catalyst only — adsorbate types are conditioned) ─────────────────────────────
        z_pred = pred["z"].float()
        z_target = data.atomic_numbers_target.long().squeeze(1) # N
        z_ce_per_atom = F.cross_entropy(z_pred, z_target, reduction="none").unsqueeze(1) # N, 1
        z_loss = (z_ce_per_atom).sum() / len(z_ce_per_atom)
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # ── Lattice losses (endpoint) ───────────────────────────────────────────────────────────────────
        len_loss = (pred["len"] - data.lengths_scaled_target).abs().mean()
        ang_loss = (pred["ang"] - data.angles_radians_target).abs().mean()
        M_loss   = (pred["M"]   - data.transformation_matrix_target).abs().mean()
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        # ── Adsorbate losses (endpoint) ─────────────────────────────────────────────────────────────────
        ads_com_loss  = (pred["ads_com"]     - data.ads_com_target).abs().mean()
        ads_relp_loss = (pred["ads_rel_pos"] - data.ads_rel_pos_target).abs().mean()
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        lambda_x   = float(self.hparams.lambda_x)
        lambda_z   = float(self.hparams.lambda_z)
        lambda_ads = float(self.hparams.lambda_ads)
        lambda_lat = float(self.hparams.lambda_lat)
        lambda_M   = float(self.hparams.lambda_M)

        total_loss = self.time_weight * (
              lambda_x   * x_loss
            + lambda_z   * z_loss
            + lambda_ads * (ads_com_loss + ads_relp_loss)
            + lambda_lat * (len_loss + ang_loss)
            + lambda_M   * M_loss
        )

        modality_losses = {
            "x_loss":        x_loss,
            "z_loss":        z_loss,
            "ads_com_loss":  ads_com_loss,
            "ads_relp_loss": ads_relp_loss,
            "len_loss":      len_loss,
            "ang_loss":      ang_loss,
            "M_loss":        M_loss,
        }
        return total_loss, modality_losses

    def path(
        self,
        input_graph: Data,
        time_dist: Literal["uniform", "beta"] = "beta",
        beta_t: float = 1.8,
    ) -> Data:
        """
        Generate interpolated training data along a path from initial to relaxed positions.

        For "refine" variant: interpolates between initial (x0) and relaxed (x1) positions
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
            self.catalyst_mask = (data.tags == 1)
            self.adsorbate_mask = (data.tags == 2)
            
            if self.variant == "forge":
                data = self._build_lattice_path(data, t=t)
                data = self._build_transformation_matrix_path(data, t=t)
                data = self._build_ads_path(data, t=t)
                data = self._build_atom_type_path(data, t=t)

            # ── Positions ────────────────────────────────────────────────────
            data = self._build_positions_path(data, t=t)

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
        t0_weight = self.hparams.t0_weight
        alpha_t_factor = torch.clamp(alpha_t_grad / (1 - alpha_t) ** 2,  min=None, max=100) if x_matching == "endpoint" \
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
        # pyg_converter stores these as (1, 3) per graph → batches to (B, 3)
        len1 = data.lengths_scaled.float()  # (B, 3)
        ang1 = data.angles_radians.float()  # (B, 3)

        len0 = torch.randn_like(len1) # B, 3
        ang0 = torch.randn_like(ang1) # B, 3

        t_len_per_graph = t # B, 1
        t_len = t_len_per_graph

        alpha_t_len, alpha_t_len_grad, alpha_t_len_loss_coeff = self._interpolant_schedule(
            t=t_len,
            schedule_type=self.schedule_type,
            x_matching=self.x_matching,
            v=cosine_schd_freq, 
        )

        t_ang_per_graph = t # B, 1
        t_ang = t_ang_per_graph

        alpha_t_ang, alpha_t_ang_grad, alpha_t_ang_loss_coeff = self._interpolant_schedule(
            t=t_ang,
            schedule_type=self.schedule_type,
            x_matching=self.x_matching,
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
        if self.x_matching == "flow":
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
            schedule_type=self.schedule_type,
            x_matching=self.x_matching,
            v=cosine_schd_freq, 
        )

        self.time_weight = alpha_t_M_loss_coeff # store one per graph loss coefficient 

        u_M = M1 - M0
        M_t = M0 + alpha_t_M.unsqueeze(-1) * u_M

        data.transformation_matrix = M_t
        data.transformation_matrix_target = M1

        # Build supercell lattice from primitive cell and transformation matrix
        # prim_cell_t: (B, 3, 3) from lattice parameters
        # M_t: (B, 3, 3) transformation matrix
        # supercell = M_t @ prim_cell_t

        # First, build primitive cell from lattice parameters
        C_t = lattice_params_to_matrix(
            lengths=data.lengths_scaled,  # (B, 3)
            angles=data.angles_radians,   # (B, 3)
        )  # (B, 3, 3

        # Then apply transformation matrix to get supercell lattice
        # Supercell lattice: L_super = M @ L_prim
        data.supercell_lattice = torch.bmm(M_t, C_t)  # (B, 3, 3)
        data.cell = C_t  # Store primitive cell as well 

        if self.x_matching == "flow":
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
        if self.variant == "place":
            if hasattr(data, "pos_relaxed"):
                x1 = data.pos_relaxed.float() # N, 3
                x0 = data.pos.float() # N, 3

                if hasattr(data, "velocity"):
                    u_x = data.velocity.float()
                else:
                    u_x = pbc_aware_delta(
                        delta=x1 - x0,
                        cell=data.cell.float(),
                        batch=data.batch,
                    )

                ads_cutoff = self.hparams.ads_cutoff
                if ads_cutoff == 0:
                    ads_neighborhood_mask = (self.adsorbate_mask)
                elif ads_cutoff == float('inf'):
                    ads_neighborhood_mask = (data.fixed == 0)
                else:
                    ads_neighborhood_mask = adsorbate_neighborhood_mask(data, ads_cutoff)

                data.ads_neighborhood_mask = ads_neighborhood_mask
                u_x = ads_neighborhood_mask.float().unsqueeze(1) * u_x

                t_x_per_graph = t

                # broadcast to per atom
                t_x = t_x_per_graph[data.batch] # N, 1

        elif self.variant == "forge":
            x1 = data.pos.float() # combined primtive cell 
            x0 = torch.randn_like(x1, dtype=torch.float32)

            if self.optimal_transport:
                print(x0.size())
                x0 = batched_ot_align(x0, x1, batch=data.batch[self.catalyst_mask], pbc_aware=True)
                print(x0.size())
            u_x = x1 - x0 
            u_x_n0 = torch.round(u_x)
            u_x = u_x - u_x_n0

            # Sample per-graph interpolation time
            t_x_per_graph = t # B, 1 

            # broadcast to per atom
            t_x = t_x_per_graph[data.batch[self.catalyst_mask]] # Nprim, 1
        
        else:
            raise ValueError("Unrecognized variant selected")

        alpha_t_x, alpha_t_x_grad, alpha_t_x_loss_coeff = self._interpolant_schedule(
            t=t_x,
            schedule_type=self.schedule_type,
            x_matching=self.x_matching,
            v=cosine_schd_freq, 
        )

        # interpolation 
        xt  = x0 + alpha_t_x * u_x       # rescaled velocity

        data.pos   = xt   
        data.pos_target = x1

        if self.x_matching == "flow":
            data.u_x = u_x * alpha_t_x_grad

        return data

    def _build_ads_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1) -> Data:
        """
        Flow-match adsorbate positions (forge variant only).

        x0 = random noise, Hungarian-aligned to x1 for permutation invariance.
        x1 = true adsorbate positions.
        xt = (1-t)*x0 + t*x1   (stays on simplex by convexity).
        Adsorbate atom types are conditioned (fixed to their true value).

        Stores on data:
            data.pos_noisy  ← argmax(xt)  (integer, for embedding)
            data.pos        ← same as above (replaces clean labels)
        """

        # pyg_converter stores ads_com as (1, 3) per graph → batches to (B, 3)
        adsCOM1 = data.ads_com.float()       # (B, 3)
        adsRELP1 = data.ads_rel_pos.float()  # (N_ads_total, 3)

        adsCOM0 = torch.randn_like(adsCOM1)   # (B, 3)
        adsRELP0 = torch.randn_like(adsRELP1) # (N_ads_total, 3)

        if self.optimal_transport:
            adsRELP0 = batched_ot_align(adsRELP0, adsRELP1, batch=data.batch[self.adsorbate_mask], pbc_aware=False)

        t_COM_per_graph = t
        t_COM = t_COM_per_graph # B, 1
        alpha_t_COM, alpha_t_COM_grad, alpha_t_COM_loss_coeff = self._interpolant_schedule(
            t=t_COM,
            schedule_type=self.schedule_type,
            x_matching=self.x_matching,
            v=cosine_schd_freq, 
        )

        t_RELP_per_graph = t # B, 1
        t_RELP = t_RELP_per_graph[data.batch[self.adsorbate_mask]] # N_ads, 1
        alpha_t_RELP, alpha_t_RELP_grad, alpha_t_RELP_loss_coeff = self._interpolant_schedule(
            t=t_RELP,
            schedule_type=self.schedule_type,
            x_matching=self.x_matching,
            v=cosine_schd_freq, 
        )

        u_COM = adsCOM1 - adsCOM0
        u_RELP = adsRELP1 - adsRELP0

        # interpolation 
        adsCOM_t = adsCOM0 + alpha_t_COM * u_COM
        adsRELP_t = adsRELP0 + alpha_t_RELP * u_RELP

        data.ads_com = adsCOM_t
        data.ads_rel_pos = adsRELP_t
        data.ads_com_target = adsCOM1
        data.ads_rel_pos_target = adsRELP1

        if self.x_matching == "flow":
            data.u_ads_com = alpha_t_COM_grad * u_COM
            data.u_ads_rel_pos = alpha_t_RELP_grad * u_RELP
        
        # See atom type path for adsorbate atom types 

        return data

    def _build_atom_type_path(self, data: Data, t: Tensor, cosine_schd_freq: Tensor=1) -> Data:
        """
        Simplex flow-matching for catalyst atom types (forge variant only).

        z0 = random one-hot, Hungarian-aligned to z1 for permutation invariance.
        z1 = true atomic numbers (one-hot).
        zt = (1-t)*z0 + t*z1   (stays on simplex by convexity).
        Adsorbate atom types are conditioned (fixed to their true value).

        Stores on data:
            data.atomic_numbers_noisy  ← argmax(zt)  (integer, for embedding)
            data.atomic_numbers        ← same as above (replaces clean labels)
            data.atomic_numbers_true   ← z1 indices (cross-entropy target)
            data.atomic_numbers_hot    ← zt  (soft distribution for soft embedding)
            data.t_z                   ← per-atom interpolation time
        """

        device = data.pos.device

        # shift atomic numbers to start from 0, not 1 
        # z1 only pertains to the primitive cell 
        z1 = (data.atomic_numbers.long() - 1).unsqueeze(1) # N, 1

        # Prior: random atomic number indices (much more efficient than one-hot)
        z0 = torch.randint(low=0, high=NUM_CLASSES, size=(z1.size(0), 1), device=device)  # N, 1

        # Hungarian alignment is intentionally skipped for atom types.
        # The corruption-mask scheme (zt = z0 * corrupt + z1 * (1-corrupt)) means
        # the model sees either the true type or a random one — the specific random
        # value in z0 doesn't affect training. Additionally, global cross-graph
        # atom matching is semantically incorrect.

        # Convert to one-hot after alignment
        # F.one_hot expects (N,) input, so squeeze the (N, 1) tensors
        z0_hot = F.one_hot(z0.squeeze(1), num_classes=NUM_CLASSES)      # N, NUM_CLASSES
        z1_hot = F.one_hot(z1.squeeze(1), num_classes=NUM_CLASSES)      # N, NUM_CLASSES
        
        # Sample per-graph interpolation time
        t_z_per_graph = t
        t_z = t_z_per_graph[data.batch[self.catalyst_mask]] # Nprim, 1

        alpha_t_z, alpha_t_z_grad, alpha_t_z_loss_coeff = self._interpolant_schedule(
            t=t_z,
            schedule_type=self.schedule_type,
            x_matching="endpoint", # atomic numbers must always use endpoint matching 
            v=cosine_schd_freq, 
        )

        # Use corruption probbability to randomly mask certain atoms 
        # Becomes less likely as t -> 1 (true sample)
        corruption_prob_per_graph = torch.rand(t.size(), device=device) # B, 1
        corruption_prob = corruption_prob_per_graph[data.batch[self.catalyst_mask]] # Nprim, 1
        corrupt_mask = (corruption_prob > t_z).int() # Nprim, 1
        zt_hot = z0_hot * corrupt_mask + z1_hot * (1 - corrupt_mask)

        data.atomic_numbers_target = z1
        data.atomic_numbers_hot = zt_hot

        # Condition on adsorbate atom types 
        z1_ads = data.ads_atomic_numbers.long() - 1  # N_ads, shift 1-90 → 0-89
        data.ads_atomic_numbers = z1_ads

        return data

    def _sample_forge(
            self,
            input_graph: Data,
            t: torch.Tensor,  # Time points, e.g. linspace(0, 1, n_steps)
            n_prim_atoms: Optional[int] = None,
    ) -> Data:
        """Generate structures for the forge variant via a custom Euler loop.

        All six modalities are integrated simultaneously:
        - Continuous (pos, lengths, angles, M, ads_com, ads_rel_pos): deterministic
          endpoint-matching Euler steps.
        - Discrete (atom types): stochastic CTMC steps using the model's predicted
          endpoint distribution, following zatom's DiscreteInterpolant.step().

        Args:
            input_graph: PyG Data/Batch. Adsorbate conditioning fields
                (ads_atomic_numbers, ads_pairwise_dist, ads_binding_energy) are
                used as-is; all generated modalities are re-initialised from prior.
            t: 1-D tensor of monotonically increasing time points in [0, 1].
                The loop takes len(t)-1 Euler steps.

        Returns:
            data with *_generated fields set for each modality.
        """
        data = input_graph.clone()
        device = data.pos.device

        self.catalyst_mask = (data.tags == 1)
        self.adsorbate_mask = (data.tags == 2)
        B = int(data.batch.max().item()) + 1

        if n_prim_atoms is not None:
            ads_mask = (data.tags == 2)
            new_prim_batch = torch.repeat_interleave(
                torch.arange(B, device=device), n_prim_atoms
            )
            new_prim_tags = torch.ones(
                B * n_prim_atoms, dtype=data.tags.dtype, device=device
            )
            data.batch = torch.cat([new_prim_batch, data.batch[ads_mask]])
            data.tags  = torch.cat([new_prim_tags,  data.tags[ads_mask]])
            data.pos   = torch.zeros(B * n_prim_atoms, 3, device=device)
            self.catalyst_mask = (data.tags == 1)
            self.adsorbate_mask = (data.tags == 2)

        # Shift ads conditioning to 0-indexed (matches _build_atom_type_path)
        data.ads_atomic_numbers = data.ads_atomic_numbers.long() - 1

        # ── Initialise all generated modalities from prior (t = 0) ───────────
        # Continuous
        data.pos = torch.randn_like(data.pos)                              # (N_slab, 3)
        data.lengths_scaled = torch.randn(B, 3, device=device)            # (B, 3)
        data.angles_radians = torch.randn(B, 3, device=device)            # (B, 3)
        identity = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        data.transformation_matrix = (
            identity + 0.1 * torch.randn(B, 3, 3, device=device)
        ).float()                                                           # (B, 3, 3)
        data.ads_com     = torch.randn_like(data.ads_com)                  # (B, 3)
        com_0 = data.ads_com
        data.ads_rel_pos = torch.randn_like(data.ads_rel_pos)              # (N_ads, 3)
        relp_0 = data.ads_rel_pos
        
        # Discrete: uniform random atom type
        N_prim = int(self.catalyst_mask.sum().item())
        z0_idx = torch.randint(0, 100, (N_prim,), device=device)
        data.atomic_numbers_hot = F.one_hot(z0_idx, num_classes=NUM_CLASSES).float()  # (N_slab, 100)

        # ── Euler integration loop ────────────────────────────────────────────
        for i in range(len(t) - 1):
            t_curr = t[i].item()
            dt     = (t[i + 1] - t[i]).item()


            # Rebuild derived lattice fields from current state
            C_t = lattice_params_to_matrix(
                lengths=data.lengths_scaled,
                angles=data.angles_radians,
            )  # (B, 3, 3)
            data.cell             = C_t
            data.supercell_lattice = torch.bmm(data.transformation_matrix, C_t)

            # Per-graph time for the model
            data.t = torch.full((B, 1), t_curr, device=device, dtype=torch.float32)

            # Model forward
            with torch.no_grad():
                pred = self._forge_forward(data)

            # Update continuous modalities
            data.pos               = euler_step(data.pos,               pred["pos"],     t_curr, dt)
            data.lengths_scaled    = euler_step(data.lengths_scaled,    pred["len"],     t_curr, dt)
            data.angles_radians    = euler_step(data.angles_radians,    pred["ang"],     t_curr, dt)
            data.transformation_matrix = euler_step(data.transformation_matrix, pred["M"], t_curr, dt)
            data.ads_com     = euler_step(data.ads_com,     pred["ads_com"],     t_curr, dt)
            data.ads_rel_pos = euler_step(data.ads_rel_pos, pred["ads_rel_pos"], t_curr, dt)

            # Update discrete atom types (CTMC)
            data.atomic_numbers_hot = ctmc_step(data.atomic_numbers_hot, pred["z"], t_curr, dt)

        # ── Store generated outputs ───────────────────────────────────────────
        data.pos_generated             = data.pos
        data.atomic_numbers_generated  = data.atomic_numbers_hot.argmax(dim=-1) + 1  # 1-indexed
        data.lengths_generated         = data.lengths_scaled
        data.angles_generated          = data.angles_radians
        data.transformation_matrix     = data.transformation_matrix.round().long()
        data.M_generated               = data.transformation_matrix
        data.ads_com_generated         = data.ads_com
        data.ads_rel_pos_generated     = data.ads_rel_pos


        return data

    def sample(
            self,
            input_graph: Data,
            t: torch.Tensor,  # Time points for ODE integration (typically [0, 1])
            ode_method: str = "dopri5",
            n_prim_atoms: Optional[int] = None,
    ) -> Data:
        """
        Generate relaxed atomic structures by integrating the learned velocity field.

        Uses an ODE solver to integrate dx/dt = u(x, t) from t=0 to t=1, where:
        - x(0) = initial positions
        - x(1) = predicted relaxed positions
        - u(x, t) = model-predicted velocity field

        Returns:
            data: Data object with pos_generated attribute containing predicted relaxed positions
        """

        if self.variant == "forge":
            return self._sample_forge(input_graph, t, n_prim_atoms=n_prim_atoms)

        data = input_graph.clone()
        x0_ref = input_graph.pos.clone().float()  # fixed reference for displacement embedding
        frozen_atoms = (data.fixed == 1)

        # Precompute spatial mask based on ads_cutoff
        ads_cutoff = self.hparams.ads_cutoff
        if ads_cutoff == 0:
            outside_neighborhood = (data.tags != 2)
        elif ads_cutoff == float('inf'):
            outside_neighborhood = frozen_atoms
        else:
            outside_neighborhood = ~adsorbate_neighborhood_mask(data, ads_cutoff)
        
        def dx_dt(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            """
            ODE right-hand side: dx/dt = u(x, t)

            Wraps the PyTorch model to match torchdiffeq's expected signature.
            """
            data.pos = xt
            data.t = t.view(1,1).expand(len(xt), -1)

            if self.x_matching == "endpoint":
                x1_hat = self(data)
                if isinstance(x1_hat, dict):
                    x1_hat = x1_hat["u_x"]
                _, _, alpha_t_factor = self._interpolant_schedule(
                    t=data.t,
                    schedule_type=self.schedule_type,
                    v=1,
                    x_matching=self.x_matching,
                )   
                delta = pbc_aware_delta(
                        delta=x1_hat - xt,
                        cell=data.cell.float(),
                        batch=data.batch,
                    )
                #alpha_t_factor = torch.clamp(alpha_t_factor, 0.005, 1.5)
                u_x =  alpha_t_factor * delta
            else: 
                u_x = self(data)  # Model predicts velocity field
                if isinstance(u_x, dict):
                    u_x = u_x["u_x"]

            # Zero velocity for atoms outside the supervised neighborhood
            u_x[outside_neighborhood] = 0

            return u_x

        # Integrate ODE from t=0 to t=1 to generate relaxed positions
        xt = odeint(
            func=dx_dt,
            y0=input_graph.pos,  # Start from initial positions
            t=t * 0.999 if self.x_matching == "endpoint" else t,  # Integration time points
            rtol=self.rtol,  # Relative tolerance
            atol=self.atol,  # Absolute tolerance
            method=ode_method,  # ODE solver method (e.g., 'dopri5', 'euler')
        )

        # Extract final positions (at t=1)
        data.pos = input_graph.pos
        data.pos_generated = xt[-1]
        
        return data 
  
