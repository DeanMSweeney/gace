import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from adsflow.utils.invariance import pbc_aware_delta, hungarian
from adsflow.utils.utils import adsorbate_neighborhood_mask

class XMatching():
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms
    """

    def criterion(
        self,
        input_graph,
        pred: Union[Tensor, dict],  # N, 3 positions or dict with {"u_x", "z_logits"}
    ):
        """
        Compute the loss between predicted and target atomic positions.

        Supports:
        - Masking for frozen atoms (only compute loss on movable atoms)
        - Optional adsorbate weighting (upweight adsorbate atoms vs catalyst)
        - PBC-aware distance computation for endpoint matching
        - Time-dependent loss weighting for cosine interpolation schedules
        """
        # Compute mask (no gradient needed for mask creation)
        data = input_graph
        pred_x = pred["u_x"] if isinstance(pred, dict) else pred
        pred_z = pred.get("z_logits", None) if isinstance(pred, dict) else None

        pred_x = pred_x.float()
        if self.x_matching == "endpoint":
            target_x = data.pos_relaxed.float()
        else:
            if not hasattr(data, "u_x"):
                raise ValueError("Flow criterion requires `data.u_x` to be set in path().")
            target_x = data.u_x.float()

        # Compute position difference based on matching mode (needs gradients!)
        if self.x_matching == "endpoint":
            # Compute PBC-aware loss between predicted and target positions
            # (handles periodic boundary conditions for crystalline structures)
            diff = pbc_aware_delta(
                cell=data.cell.float(),
                delta=(pred_x - target_x),
                batch=data.batch,
            )

        elif self.x_matching == "flow":
            # Simple Euclidean difference (for flow-based methods
            diff = pred_x - target_x

        else:
            raise ValueError("Invalid x_matching routine")

        with torch.no_grad():
            # Apply time-dependent weighting for endpoint interpolation schedules.
            # During forge training, alpha_t_x_factor is set in path().
            alpha_t_x_factor = getattr(self, "alpha_t_x_factor", 1.0)

            # Spatial mask (precomputed in path(), stored on data)
            if hasattr(data, "ads_neighborhood_mask"):
                mask = data.ads_neighborhood_mask.float().unsqueeze(1)
            else:
                mask = (data.fixed == 0).float().unsqueeze(1)

            # Adsorbate weighting within the selected atoms
            tao = self.hparams.tao
            adsorbates = (data.tags == 2).float().unsqueeze(1)  # N, 1
            mask_factor = mask * (1 + adsorbates * (tao - 1))

        # Compute mean squared error per atom (averaged over xyz dimensions)
        L_per_atom = diff.abs().mean(dim=1, keepdim=True)  # N, 1
        
        # Apply all weighting factors: frozen atom mask, adsorbate weight, time weight.
        modified_L_per_atom = L_per_atom * mask_factor * alpha_t_x_factor  # N, 1
        # Average over all free (non-frozen) atoms
        x_loss = modified_L_per_atom.sum() / mask.sum().clamp_min(1e-8)
        
        with torch.no_grad():
            # Track maximum loss atom for stability monitoring
            x_max_L = modified_L_per_atom.max()

        # Forge adds atomic-number endpoint loss.
        if self.variant == "relax":
            return x_loss, x_max_L

        if pred_z is None:
            raise ValueError("Forge criterion requires predictions dict with `z_logits`.")
        if not hasattr(data, "atomic_numbers_true"):
            raise ValueError("Forge criterion requires `data.atomic_numbers_true`.")

        z_targets = data.atomic_numbers_true.long()
        z_ce_per_atom = F.cross_entropy(pred_z.float(), z_targets, reduction="none").unsqueeze(1)

        # Adsorbate types are conditioned/fixed in forge; supervise catalyst types.
        z_mask = (data.tags != 2).float().unsqueeze(1)
        z_loss = (z_ce_per_atom * z_mask).sum() / z_mask.sum().clamp_min(1.0)

        lambda_z = float(getattr(self.hparams, "lambda_z", 1.0))
        total_loss = x_loss + lambda_z * z_loss

        with torch.no_grad():
            z_max_L = (lambda_z * z_ce_per_atom * z_mask).max()
            max_L = torch.maximum(x_max_L, z_max_L)
        
        return total_loss, max_L

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
        alpha_t_factor = torch.clamp(alpha_t_grad / (1 - alpha_t), 0.005, 1.5) if x_matching == "endpoint" \
            else torch.clamp(1 / t, 1.0, t0_weight) #torch.tensor(1.0, device=t.device, dtype=torch.float32)

        return alpha_t, alpha_t_grad, alpha_t_factor
            
    def path(
        self,
        input_graph: Data,
        v_x: float=1,  # exponent for cosine schedule (controls interpolation curvature)
        v_z: float=1,  # exponent for cosine schedule (controls interpolation curvature)
        initial_dist: Literal["noisy", "fixed"] = "fixed",
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

            # ── Lattice and supercell (CatFlow data only) ────────────────────
            if hasattr(data, "prim_cell"):
                data = self._apply_lattice_flow(data)
            elif hasattr(data, "supercell_matrix"):
                data = self._apply_supercell_flow(data)
            else:
                raise ValueError("Data must have either 'prim_cell' and 'supercell_matrix'")

            # ── Positions ────────────────────────────────────────────────────
            data = self._apply_positions_flow(data, v_x)

            if self.variant == "place":
                return data

            # ── Atomic numbers (forge only) ──────────────────────────────────
            data = self._apply_atomic_num_flow(data, v_z)

            return data

        raise ValueError("Invalid model variant")


    # ─────────────────────────────────────────────────────────────────────────
    # Modular flow helpers — each takes a Data object and returns it updated.
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_lattice_flow(self, data: Data) -> Data:
        """
        Flow-match the primitive cell matrix (3×3 Cartesian, Å).

        x0  ~ N(0, I)  scaled to match the RMS magnitude of the true cell.
        x1  = data.prim_cell  (true primitive cell, shape (3,3) or (B,3,3))
        xt  = x0 + alpha_t * (x1 - x0)   (schedule-interpolated)
        u_L = alpha_t_grad * (x1 - x0)   (rescaled target velocity)

        The noisy cell `xt` is stored back as `data.prim_cell` so that
        downstream graph construction uses it at the current flow time.
        """
        device = data.prim_cell.device
        dtype  = torch.float32

        L1 = data.prim_cell.float()                       # (..., 3, 3)

        # Prior: zero-mean Gaussian scaled to RMS magnitude of a typical cell
        L0 = torch.randn_like(L1) * L1.norm() / (3 * 3) ** 0.5

        # Per-graph time sample — shape (1,) broadcasts over (3,3)
        t_L = torch.rand(1, device=device, dtype=dtype)

        alpha_t_L, alpha_t_L_grad, _ = self._interpolant_schedule(
            t=t_L,
            schedule_type=self.schedule_type,
            x_matching="flow",
        )

        u_L    = L1 - L0
        cell_t = L0 + alpha_t_L * u_L
        u_L    = alpha_t_L_grad * u_L  # rescaled velocity

        data.prim_cell        = cell_t
        data.prim_cell_target = L1
        data.u_L              = u_L
        data.t_L              = t_L

        return data

    def _apply_supercell_flow(self, data: Data) -> Data:
        """
        Flow-match the supercell matrix M  (3×3, integer-valued targets).

        x0  ~ I + 0.1·N(0,1)  (prior near-identity; supercells are usually close to I)
        x1  = data.supercell_matrix  (true integer supercell matrix)
        xt  = x0 + alpha_t * (x1 - x0)   (schedule-interpolated)
        u_M = alpha_t_grad * (x1 - x0)   (rescaled target velocity)

        Cosine regularisation `cosine_reg_M` encourages integer predictions:
            reg = mean(1 - cos(2π * M_pred))   (minimised at integers)
        Add lambda_M * data.cosine_reg_M to the total loss in criterion().
        """
        device = data.supercell_matrix.device
        dtype  = torch.float32

        M1 = data.supercell_matrix.float()                   # (..., 3, 3)

        identity = torch.eye(3, device=device, dtype=dtype).expand_as(M1)
        M0 = identity + 0.1 * torch.randn_like(M1)

        t_M = torch.rand(1, device=device, dtype=dtype)

        alpha_t_M, alpha_t_M_grad, _ = self._interpolant_schedule(
            t=t_M,
            schedule_type=self.schedule_type,
            x_matching="flow",
        )

        u_M = M1 - M0
        M_t = M0 + alpha_t_M * u_M
        u_M = alpha_t_M_grad * u_M                  # rescaled velocity

        # Cosine regularisation template (evaluated on model output in criterion)
        cosine_reg_M = (1.0 - torch.cos(2 * torch.pi * M1)).mean()

        data.supercell_matrix        = M_t
        data.supercell_matrix_target = M1
        data.u_M                     = u_M
        data.t_M                     = t_M
        data.cosine_reg_M            = cosine_reg_M
        data.cell_target = data.cell 
        data.cell = data.prim_cell @ data.supercell_matrix

        return data

    def _apply_positions_flow(self, data: Data, v_x: float = 1) -> Data:
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

        if hasattr(data, "pos_relaxed"):
            x1 = data.pos_relaxed.float()
        elif hasattr(data, "pos_prim"):
            x1 = data.pos_combined.float() # combined primtive cell + adsorbate
        else:
            raise ValueError("Data must have 'pos_relaxed' or 'pos_prim' attribute")

        if self.variant == "relax":
            x0 = data.pos.float()

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
                ads_neighborhood_mask = (data.tags == 2)
            elif ads_cutoff == float('inf'):
                ads_neighborhood_mask = (data.fixed == 0)
            else:
                ads_neighborhood_mask = adsorbate_neighborhood_mask(data, ads_cutoff)

            data.ads_neighborhood_mask = ads_neighborhood_mask
            u_x = ads_neighborhood_mask.float().unsqueeze(1) * u_x

        elif self.variant == "forge":
            adsorbate = (data.tags == 2)
            catalyst  = (data.tags != 2)

            x0 = torch.randn_like(x1, dtype=torch.float32)
            # Both slab and adsorbate start from Gaussian noise.
            # Hungarian alignment applied to catalyst atoms only —
            # adsorbate atoms have fixed chemical identity and must not be permuted.
            x0[catalyst] = hungarian(
                set0=x0[catalyst],
                set1=x1[catalyst],
                cell=data.cell.float(),
                batch=data.batch,
            )

            u_x = pbc_aware_delta(
                delta=x1 - x0,
                cell=data.cell.float(),
                batch=data.batch,
            )
        else:
            raise ValueError(f"Unknown variant '{self.variant}' in _apply_positions_flow")

        # Sample per-graph interpolation time
        t_per_graph = torch.rand(
            len(data.natoms), 1, device=x1.device, dtype=torch.float32
        )
        t_x = t_per_graph[data.batch] * 0.999 if self.x_matching == "endpoint" \
              else t_per_graph[data.batch]

        alpha_t_x, alpha_t_x_grad, self.alpha_t_x_factor = self._interpolant_schedule(
            t=t_x,
            schedule_type=self.schedule_type,
            v=v_x,
            x_matching="flow",
        )

        xt  = x0 + alpha_t_x * u_x
        u_x = alpha_t_x_grad * u_x          # rescaled velocity

        data.pos      = xt
        data.pos_init = x0
        data.t        = t_x
        data.u_x      = u_x

        return data

    def _apply_atomic_num_flow(self, data: Data, v_z: float = 1) -> Data:
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
        adsorbate = (data.tags == 2)
        catalyst  = (data.tags != 2)
        device    = data.pos.device

        num_classes = 90  # matches max_num_elements in EquiformerV2
        z1_raw = data.atomic_numbers.long()
        if z1_raw.min() >= 1 and z1_raw.max() <= num_classes:
            z1 = z1_raw - 1          # shift chemical numbers to 0-based
        elif z1_raw.min() >= 0 and z1_raw.max() < num_classes:
            z1 = z1_raw
        else:
            raise ValueError(
                f"atomic_numbers out of range: [{int(z1_raw.min())}, {int(z1_raw.max())}]"
            )

        z1_hot = F.one_hot(z1, num_classes=num_classes).float()      # (N, 90)

        # Prior: random one-hot per atom
        z0_hot = torch.zeros_like(z1_hot)
        rand_idx = torch.randint(0, num_classes, (z0_hot.size(0),), device=device)
        z0_hot[torch.arange(z0_hot.size(0), device=device), rand_idx] = 1.0

        # Adsorbate types are conditioned — fix to ground truth
        z0_hot[adsorbate] = z1_hot[adsorbate]

        # Hungarian alignment over catalyst for permutation invariance
        z0_hot[catalyst] = hungarian(
            set0=z0_hot[catalyst],
            set1=z1_hot[catalyst],
            batch=data.batch[catalyst],
        )

        t_z_per_graph = torch.rand(len(data.natoms), 1, device=device, dtype=torch.float32)
        t_z = t_z_per_graph[data.batch] * 0.999

        alpha_t_z, _, _ = self._interpolant_schedule(
            t=t_z,
            schedule_type=self.schedule_type,
            x_matching="endpoint",
        )

        zt_hot = (1 - alpha_t_z) * z0_hot + alpha_t_z * z1_hot      # (N, 90)

        data.atomic_numbers_hot   = zt_hot
        data.atomic_numbers       = zt_hot.argmax(dim=-1).long()
        data.atomic_numbers_true  = z1
        data.t_z                  = t_z
        
        return data

    def sample(
            self,
            input_graph: Data,
            t: torch.Tensor,  # Time points for ODE integration (typically [0, 1])
            ode_method: str = "dopri5",
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
  
