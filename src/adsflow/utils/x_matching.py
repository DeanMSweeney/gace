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

            # x1 = final relaxed positions (target)
            x1 = data.pos_relaxed.float()

             # REFINE variant: refine from initial DFT positions to relaxed positions
            if initial_dist == "fixed":
                x0 = data.pos.float() # Initial (unrelaxed) positions
                
                if hasattr(data, "velocity"):
                    u_x = data.velocity.float() # x1 - x0 
                else:
                    u_x = pbc_aware_delta(
                        delta=x1 - x0,
                        cell=data.cell.float(),
                        batch=data.batch,
                    )

                # Determine which atoms to supervise based on ads_cutoff
                ads_cutoff = self.hparams.ads_cutoff
                if ads_cutoff == 0:
                    # Adsorbate atoms only
                    ads_neighborhood_mask = (data.tags == 2)
                elif ads_cutoff == float('inf'):
                    # All non-frozen atoms
                    ads_neighborhood_mask = (data.fixed == 0)
                else:
                    # Adsorbate + catalyst within cutoff
                    ads_neighborhood_mask = adsorbate_neighborhood_mask(data, ads_cutoff)
                
                data.ads_neighborhood_mask = ads_neighborhood_mask  # store for criterion()
                path_mask = ads_neighborhood_mask.float().unsqueeze(1)
                u_x = path_mask * u_x 
                    

            elif initial_dist == "noisy":
                adsorbate = (data.tags == 2)
                catalyst = (data.tags != 2)

                # x0 = initial positions (noise for catalyst, actual for adsorbate)
                x0 = torch.randn_like(x1, dtype=torch.float32)
                x0[adsorbate] = data.pos[adsorbate].float() # Use actual adsorbate positions

                # Use Hungarian algorithm to optimally align catalyst atoms
                # (finds permutation that minimizes distance to target)
                x0[catalyst] = hungarian(
                    set0=x0[catalyst],
                    set1=x1[catalyst],
                    cell=data.cell.float(),
                    batch=data.batch,
                )

                delta_xyz = x1 - x0
                u_x = pbc_aware_delta(
                    delta=delta_xyz,
                    cell=data.cell.float(),
                    batch=data.batch,
                )

            # Sample random interpolation time t for each graph in the batch
            t_x_per_graph = torch.rand(len(data.natoms), 1, device=x1.device, dtype=torch.float32)
            # Broadcast to atoms, clamp max to 0.999
            t_x = t_x_per_graph[data.batch] * 0.999 if self.x_matching == "endpoint" else t_x_per_graph[data.batch]
            # Compute interpolation schedule parameters
            alpha_t_x, alpha_t_x_grad, self.alpha_t_x_factor = self._interpolant_schedule(
                t=t_x,
                schedule_type=self.schedule_type,
                v=v_x,
                x_matching=self.x_matching,
            )

            # Interpolate: xt = x0 + alpha_t * u_x(position at time t)
            xt = x0 + alpha_t_x * u_x 

            # Rescale velocity for cosine schedule (to account for d(alpha_t)/dt)
            u_x = alpha_t_x_grad * u_x # alpha_t' * (x1 - x0)         

            # Store interpolated data
            data.pos = xt  # Current position (what model sees as input)
            data.pos_init = x0  # Initial position (for reference)
            data.t = t_x  # Interpolation time
            data.u_x = u_x
            
            if self.variant == "relax":
                return data

            elif self.variant == "forge":
                """
                Atomic number endpoint matching on the simplex (SimplexFlow).

                Catalyst atom types are discrete labels, lifted onto the
                probability simplex via one-hot encoding.  The model sees an
                interpolated soft distribution zt_hot as input and predicts
                the true element z1 directly (logits → softmax → simplex).
                Loss: cross-entropy against the true label z1.

                This endpoint parameterization guarantees that generated
                trajectories remain on the simplex by construction.
                """
                # Define adsorbate/catalyst masks (may not exist if initial_dist == "fixed")
                adsorbate = (data.tags == 2)
                catalyst  = (data.tags != 2)

                # --- Target: true atomic numbers ---
                num_classes = 90  # matches max_num_elements in EquiformerV2
                z1_raw = data.atomic_numbers.long()                           # (N,)
                # Accept either chemical atomic numbers (1..K) or zero-based indices (0..K-1).
                if z1_raw.min() >= 1 and z1_raw.max() <= num_classes:
                    z1 = z1_raw - 1
                elif z1_raw.min() >= 0 and z1_raw.max() < num_classes:
                    z1 = z1_raw
                else:
                    raise ValueError(
                        f"atomic_numbers out of range for num_classes={num_classes}: "
                        f"[{int(z1_raw.min())}, {int(z1_raw.max())}]"
                    )
                z1_hot = F.one_hot(z1, num_classes=num_classes).float()        # (N, 90)

                # --- Initial: random one-hot noise for catalyst atoms ---
                z0_hot = torch.zeros_like(z1_hot)                             # (N, 90)
                z0_rand_int = torch.randint(
                    low=0, high=num_classes,
                    size=(z0_hot.size(0),), device=x1.device,
                )
                z0_hot[torch.arange(z0_hot.size(0), device=x1.device), z0_rand_int] = 1.0

                # Adsorbate atoms keep their true type (conditioning)
                z0_hot[adsorbate] = z1_hot[adsorbate]

                # Hungarian matching for permutation invariance over catalyst atoms
                z0_hot[catalyst] = hungarian(
                    set0=z0_hot[catalyst],
                    set1=z1_hot[catalyst],
                    batch=data.batch[catalyst],
                )

                # --- Sample interpolation time (independent of position time) ---
                t_z_per_graph = torch.rand(
                    len(data.natoms), 1, device=x1.device, dtype=torch.float32,
                )
                t_z = t_z_per_graph[data.batch] * 0.999

                # Compute interpolation schedule
                alpha_t_z, _, _= self._interpolant_schedule(
                    t=t_z,
                    schedule_type=self.schedule_type,
                    v=v_z,
                    x_matching=self.x_matching,
                )

                # Interpolated soft distribution at time t (stays on the simplex)
                zt_hot = (1 - alpha_t_z) * z0_hot + alpha_t_z * z1_hot       # (N, 90)

                # Keep noisy hard labels for modules that still require integer indexing.
                data.atomic_numbers_noisy = zt_hot.argmax(dim=-1).long()      # (N,) noisy element
                data.atomic_numbers = data.atomic_numbers_noisy
                data.atomic_numbers_true = z1                                 # (N,) true labels (for CE loss)
                data.atomic_numbers_hot  = zt_hot                             # soft distribution (for soft embedding)
                data.t_z = t_z

                """
                Unit cell flow matching — TODO
                """

                return data
  
            else:
                raise ValueError("Invalid model variant")


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
  
