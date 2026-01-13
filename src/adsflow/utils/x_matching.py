import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple

from adsflow.utils.post_processing import config_DFT, write_strucs
from adsflow.utils.invariance import pbc_aware_delta, hungarian
from adsflow.utils.eval_metrics import ADwT

class XMatching(): 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
    def criterion(
        self,
        input_data,
        a: Tensor,  # N, 3 - predicted positions
        b: Tensor,  # N, 3 - target positions
        tao: float=0,  # adsorbate weighting factor (if > 0, upweight adsorbate atoms)
    ):
        """
        Compute the loss between predicted and target atomic positions.

        Supports:
        - Masking for frozen atoms (only compute loss on movable atoms)
        - Optional adsorbate weighting (upweight adsorbate atoms vs catalyst)
        - PBC-aware distance computation for endpoint matching
        - Time-dependent loss weighting for cosine interpolation schedules
        """
        # Force fp32 precision for all operations
        a = a.float()
        b = b.float()

        # Compute mask (no gradient needed for mask creation)
        data = input_data
        with torch.no_grad():
            data.fixed = data.fixed.float()
            # Create mask: 0 for frozen atoms, 1 for free atoms (N, 1)
            mask = (1 - data.fixed).unsqueeze(1)

        # Compute position difference based on matching mode (needs gradients!)
        if self.x_matching == "endpoint":
            # Compute PBC-aware loss between predicted and target positions
            # (handles periodic boundary conditions for crystalline structures)
            diff = pbc_aware_delta(
                cell=data.cell.float(),
                delta=(a - b),
                batch=data.batch,
            )
        elif self.x_matching == "flow":
            # Simple Euclidean difference (for flow-based methods)
            diff = a - b
        else:
            raise ValueError("Invalid x_matching routine")

        # Apply adsorbate weighting if tao > 0 (needs gradients!)
        if tao > 0:
            # Identify adsorbate atoms (tags == 2)
            with torch.no_grad():
                ads = (data.tags == 2).float().unsqueeze(1)  # N, 1
            # Upweight adsorbate atoms by factor tao: mask * (1 + ads * (tao - 1))
            # e.g., if tao=2, adsorbate atoms weighted 2x more than catalyst
            ads_mask_factor = mask * (1 + ads * (tao - 1))
        else:
            ads_mask_factor = mask

        # Compute mean squared error per atom (averaged over xyz dimensions)
        L_per_atom = (diff ** 2).mean(dim=1, keepdim=True)  # N, 1

        # Apply time-dependent weighting for cosine interpolation schedules
        # (helps stabilize training by reweighting loss based on interpolation time)
        time_factor = torch.clamp(self.alpha_t_factor.float(), 0.005, 1.5) if self.interpolant_schedule == "cosine" \
             else torch.tensor(1.0, device=L_per_atom.device, dtype=torch.float32)

        # Apply all weighting factors: frozen atom mask, adsorbate weight, time weight
        weighted_L_per_atom = L_per_atom * ads_mask_factor * time_factor  # N, 1
        # Average over all free (non-frozen) atoms
        L = weighted_L_per_atom.sum() / mask.sum().clamp_min(1e-8)

        with torch.no_grad():
            # Track maximum loss atom for stability monitoring
            max_L = weighted_L_per_atom.max()
            max_L_idx = torch.argmax(weighted_L_per_atom)

        return L, max_L, max_L_idx
            
    def path(
        self,
        input_data: Data,
        interpolant_schedule: str="linear",  # "linear" or "cosine" interpolation
        v: float=1,  # exponent for cosine schedule (controls interpolation curvature)
    ) -> Tuple[Data, Tensor]:
        """
        Generate interpolated training data along a path from initial to relaxed positions.

        For "refine" variant: interpolates between initial (x0) and relaxed (x1) positions
        For "forge" variant: interpolates from noise/initial positions to relaxed positions

        Returns:
            data: Modified data object with interpolated positions at time t
            u: Target velocity field (dx/dt) for the model to predict
        """

        with torch.no_grad():
            data = input_data.clone()

            # x1 = final relaxed positions (target)
            x1 = data.pos_relaxed.float()

            # Sample random interpolation time t for each graph in the batch
            t_per_graph = torch.rand(len(data.natoms), 1, device=x1.device, dtype=torch.float32)
            t = t_per_graph[data.batch] * 0.999  # Broadcast to atoms, clamp max to 0.999

            # Define interpolation schedule: how to move from t to alpha_t
            if interpolant_schedule == "linear":
                # Linear interpolation: alpha_t = t
                alpha_t = t
                self.alpha_t_factor = torch.tensor(1.0, device=t.device, dtype=torch.float32)
            elif interpolant_schedule == "cosine":
                # Cosine interpolation: slower at start/end, faster in middle
                # alpha_t = 1 - cos^2(t^v * pi/2)
                alpha_t = 1 - (torch.cos(t ** v * torch.pi / 2)) ** 2
                # Compute d(alpha_t)/dt for velocity rescaling
                alpha_t_grad = torch.pi * v * t ** (v - 1) * torch.cos(t ** v * torch.pi / 2) * torch.sin(t ** v * torch.pi / 2)
                # Store for time-dependent loss weighting
                self.alpha_t_factor = alpha_t_grad / (1 - alpha_t)
            else:
                raise ValueError("Invalid interpolant schedule")

            if self.variant == "forge":
                    # FORGE variant: generate structures from noise + adsorbate positions
                    # Partition atoms into adsorbate (tags==2) and catalyst (tags!=2)
                    adsorbate = (data.tags == 2)
                    catalyst = (data.tags != 2)

                    # x0 = initial positions (noise for catalyst, actual for adsorbate)
                    x0 = torch.randn_like(x1, dtype=torch.float32)
                    x0[adsorbate] = data.pos[adsorbate].float()  # Use actual adsorbate positions

                    # Use Hungarian algorithm to optimally align catalyst atoms
                    # (finds permutation that minimizes distance to target)
                    x0[catalyst] = hungarian(
                        s0=x0[catalyst],
                        s1=x1[catalyst],
                        cell=data.cell.float(),
                        batch=data.batch,
                    )

                    # Sample noise for atomic numbers (for conditional generation)
                    z1 = data.atomic_number
                    z1_hot = F.one_hot(z1, num_classes=55)
                    z0 = torch.rand_like(z1_hot, dtype=torch.float32)

            elif self.variant == "refine":
                # REFINE variant: refine from initial DFT positions to relaxed positions
                x0 = data.pos.float()  # Initial (unrelaxed) positions

                if hasattr(data, "velocity"):
                    # Use precomputed velocity if available (e.g., from optimal transport)
                    u = data.velocity.float()
                else:
                    # Compute PBC-aware velocity: u = x1 - x0 (with periodic boundaries)
                    delta_xyz = x1 - x0
                    u = pbc_aware_delta(
                        delta=delta_xyz,
                        cell=data.cell.float(),
                        batch=data.batch,
                    )

                # Interpolate: xt = x0 + alpha_t * u (position at time t)
                xt = x0 + alpha_t * u

                # Rescale velocity for cosine schedule (to account for d(alpha_t)/dt)
                if interpolant_schedule == "cosine":
                    self.u = alpha_t_grad * u
                else:
                    self.u = u

                # Store interpolated data
                data.pos = xt  # Current position (what model sees as input)
                data.pos_init = x0  # Initial position (for reference)
                data.t = t  # Interpolation time
            else:
                raise ValueError("Invalid model variant")

        return data, self.u 
  
    def sample(
            self,
            input_data: Data,
            t: torch.Tensor,  # Time points for ODE integration (typically [0, 1])
            threshold: float=0.20,  # Distance threshold for ADwT metric (Angstroms)
    ) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """
        Generate relaxed atomic structures by integrating the learned velocity field.

        Uses an ODE solver to integrate dx/dt = u(x, t) from t=0 to t=1, where:
        - x(0) = initial positions
        - x(1) = predicted relaxed positions
        - u(x, t) = model-predicted velocity field

        Returns:
            If ground truth available: (data, gen_mae, gen_dwt, gen_adwt, init_mae, init_dwt, init_adwt)
            Otherwise: data
        """

        data = input_data.clone()

        def dx_dt(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            """
            ODE right-hand side: dx/dt = u(x, t)

            Wraps the PyTorch model to match torchdiffeq's expected signature.
            """
            data.pos = xt
            data.t = t.view(1,1).expand(len(xt), -1)
            u = self(data)  # Model predicts velocity field
            return u

        # Integrate ODE from t=0 to t=1 to generate relaxed positions
        xt = odeint(
            func=dx_dt,
            y0=input_data.pos,  # Start from initial positions
            t=t,  # Integration time points
            rtol=self.rtol,  # Relative tolerance
            atol=self.atol,  # Absolute tolerance
            method=self.ode_method,  # ODE solver method (e.g., 'dopri5', 'euler')
        )

        # Extract final positions (at t=1)
        data.pos = input_data.pos
        data.pos_generated = xt[-1]

        # Enforce frozen atoms: overwrite generated positions with initial positions
        frozen_atoms = (data.fixed == 1)
        data.pos_generated[frozen_atoms] = data.pos[frozen_atoms]

        # Write generated structures to files if output paths provided
        if self.gen_POS:
                write_strucs(
                    data=data,
                    gen_POS=self.gen_POS,  # Generated structure path
                    in_POS=self.in_POS,  # Initial structure path
                    rlxd_POS=self.rlxd_POS,  # Relaxed structure path
                )

                # Optionally configure VASP input files for DFT verification
                if self.config_vasp:
                    config_DFT(
                        data=data,
                        gen_POS=self.gen_POS,
                        in_POS=self.in_POS,
                        rlxd_POS=self.rlxd_POS,
                    )
        else:
            print("Output file not provided... structure will not be saved")

        # Compute evaluation metrics if ground truth relaxed positions available
        if hasattr(data, "pos_relaxed"):

            # Prepare data for metric computation
            mask = (1 - data.fixed).bool()  # Mask for free (non-frozen) atoms
            cell = data.cell
            batch = data.batch
            pos_init = data.pos
            pos_predicted = data.pos_generated
            pos_target = data.pos_relaxed

            # Compute metrics for generated structure:
            # - MAE: Mean Absolute Error
            # - DwT: Distance within Threshold (% atoms within threshold)
            # - ADwT: Adsorbate Distance within Threshold
            generated_mae, generated_dwt, generated_adwt = ADwT(
               pos_predicted=pos_predicted,
               pos_target=pos_target,
               threshold=threshold,
               fixed_atoms_mask=mask,
               cell=cell,
               batch=batch,
           )

            # Compute baseline metrics for initial structure (before refinement)
            initial_mae, initial_dwt, initial_adwt = ADwT(
               pos_predicted=pos_init,
               pos_target=pos_target,
               threshold=threshold,
               fixed_atoms_mask=mask,
               cell=cell,
               batch=batch,
           )
            return data, generated_mae, generated_dwt, generated_adwt, initial_mae, initial_dwt, initial_adwt
        else:
            return data
  