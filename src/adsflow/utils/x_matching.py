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
        tao=torch.tensor(tao, device=a.device, dtype=torch.float32)

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
        time_factor = torch.clamp(self.alpha_t_factor, 0.005, 1.5) if self.schedule_type== "cosine" \
             else self.alpha_t_factor

        # Apply all weighting factors: frozen atom mask, adsorbate weight, time weight
        weighted_L_per_atom = L_per_atom * ads_mask_factor * time_factor  # N, 1
        # Average over all free (non-frozen) atoms
        L = weighted_L_per_atom.sum() / mask.sum().clamp_min(1e-8)

        with torch.no_grad():
            # Track maximum loss atom for stability monitoring
            max_L = weighted_L_per_atom.max()
            max_L_idx = torch.argmax(weighted_L_per_atom)

        return L, max_L, max_L_idx

    def _interpolant_schedule(
        self,
        t: Tensor,
        schedule_type: str,
        v_x: float = 1,
        v_z: float = 1,
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
            alpha_t_factor = 1 / (1 - t) if self.x_matching == "endpoint" else torch.tensor(1.0, device=t.device, dtype=torch.float32)

        elif schedule_type == "cosine":
            # Cosine interpolation: slower at start/end, faster in middle
            # alpha_t = 1 - cos^2(t^v * pi/2)
            alpha_t = 1 - (torch.cos(t ** v_x * torch.pi / 2)) ** 2
            # Compute d(alpha_t)/dt for velocity rescaling
            alpha_t_grad = torch.pi * v_x * t ** (v_x - 1) * torch.cos(t ** v_x * torch.pi / 2) * torch.sin(t ** v_x * torch.pi / 2)
            # Store for time-dependent loss weighting
            alpha_t_factor = alpha_t_grad / (1 - alpha_t).clamp_min(1e-8) # alpha_t'/ (1-alpha)
        else:
            raise ValueError("Invalid interpolant schedule")

        return alpha_t, alpha_t_grad, alpha_t_factor
            
    def path(
        self,
        input_data: Data,
        v_x: float=1,  # exponent for cosine schedule (controls interpolation curvature)
        v_z: float=1,  # exponent for cosine schedule (controls interpolation curvature)
        initial_dist: Literal["noisy", "fixed"] = "fixed",
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
                v_x=v_x,
            )

            # Interpolate: xt = x0 + alpha_t * u_x(position at time t)
            xt = x0 + alpha_t_x * u_x

            # Rescale velocity for cosine schedule (to account for d(alpha_t)/dt)
            u_x = alpha_t_x_grad * u_x# alpha_t' * (x1 - x0)

            # Store interpolated data
            data.pos = xt  # Current position (what model sees as input)
            data.pos_init = x0  # Initial position (for reference)
            data.t = t_x  # Interpolation time

            """
            # convert u_x to fractional coordinates
            u_x = pbc_aware_delta(
                delta=u_x,
                cell=data.cell.float(),
                batch=data.batch,
                cartesian_to_fractional=True, 
            )
            """
            if self.variant == "relax":
                return data, u_x

            elif self.variant == "forge":
                # Sample  atomic numbers (for conditional generation)
                z1 = data.atomic_number # N 

                z1_hot = F.one_hot(z1, num_classes=55) # N , 55
                
                # Sample noise by defining initial one-hot-encoded vectors 
                z0_hot = torch.zeros_like(z1_hot, dtype=torch.float32) # N , 55
                z0_row_idx = torch.arange(z0_hot.size(0), device=x1.device)
                z0_rand_int = torch.randint(low=0, high=55, size=(z0_hot.size(0),), device=x1.device) # N
                z0_hot[z0_row_idx, z0_rand_int] = 1 # randomly replace one element in each row with 1 
                z0_hot[adsorbate] = z1_hot[adsorbate] # reassign adsorabte atom types to match target

                z0_hot[catalyst] = hungarian(
                    set0=z0_hot[catalyst],
                    set1=z1_hot[catalyst],
                    batch=data.batch,
                )

                # compute velocity field for atomic numbers 
                u_z = z1_hot - z0_hot

                # Sample random interpolation time t for each graph in the batch
                t_z_per_graph = torch.rand(len(data.natoms), 1, device=x1.device, dtype=torch.float32)
                # Broadcast to atoms, clamp max to 0.999
                t_z = t_z_per_graph[data.batch] * 0.999 if self.x_matching == "endpoint" else t_z_per_graph[data.batch]
                # Compute interpolation schedule parameters
                alpha_t_z, alpha_t_z_grad, self.alpha_t_z_factor = self._interpolant_schedule(
                    t=t_z,
                    schedule_type=self.schedule_type,
                    v_z=v_z,
                )

                zt_hot = z0_hot + alpha_t_z * (z1_hot - z0_hot)

                # Rescale velocity for cosine schedule (to account for d(alpha_t)/dt)
                u_z = alpha_t_z_grad * u_z # alpha_t' * (x1 - x0)

                # Store interpolated data
                data.atomic_number = zt_hot  # Current position (what model sees as input)
                data.atomic_number_init = z0_hot  # Initial position (for reference)
                data.t_z = t_z  # Interpolation time

                return data, u_x, u_z
  
            else:
                raise ValueError("Invalid model variant")


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

            if self.x_matching == "endpoint":
                x1_hat = self(data)
                _, _, alpha_t_factor = self._interpolant_schedule(
                    t=data.t,
                    schedule_type=self.schedule_type,
                    v_x=1,
                )   
                delta = pbc_aware_delta(
                        delta=x1_hat - xt,
                        cell=data.cell.float(),
                        batch=data.batch,
                    )
                #alpha_t_factor = torch.clamp(alpha_t_factor, 0.005, 1.5)
                u_x=  alpha_t_factor * delta
            else: 
                u_x= self(data)  # Model predicts velocity field
            return u

        # Integrate ODE from t=0 to t=1 to generate relaxed positions
        xt = odeint(
            func=dx_dt,
            y0=input_data.pos,  # Start from initial positions
            t=t * 0.999 if self.x_matching == "endpoint" else t,  # Integration time points
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
            # Compute metrics for generated structure:
            # - MAE: Mean Absolute Error
            # - DwT: Distance within Threshold (% atoms within threshold)
            # - ADwT: Adsorbate Distance within Threshold
            # Compute metrics for generated structure:

            # compute velocity field on input struc for logging
            u_x= data.pos_relaxed - data.pos
            u_hat = self(data)

            gen_metrics = ADwT(data=data, pos_predicted=data.pos_generated, threshold=threshold)
            init_metrics = ADwT(data=data, pos_predicted=data.pos, threshold=threshold)

            # Return metrics as a dictionary for cleaner handling
            eval_metrics = {
                "generated_mae": gen_metrics[0],
                "generated_dwt": gen_metrics[1],
                "generated_adwt": gen_metrics[2],
                "generated_ads_mae": gen_metrics[3],
                "generated_ads_dwt": gen_metrics[4],
                "generated_ads_adwt": gen_metrics[5],
                "generated_me": gen_metrics[6],
                "generated_ads_me": gen_metrics[7],
                "initial_mae": init_metrics[0],
                "initial_dwt": init_metrics[1],
                "initial_adwt": init_metrics[2],
                "initial_ads_mae": init_metrics[3],
                "initial_ads_dwt": init_metrics[4],
                "initial_ads_adwt": init_metrics[5],
                "initial_me": init_metrics[6],
                "initial_ads_me": init_metrics[7],
                "u": u.detach().cpu().numpy(),  # Nx3 ground truth velocity field
                "u_hat": u_hat.detach().cpu().numpy(),  # Nx3 predicted velocity field
            }

            return data, eval_metrics
        else:
            return data
  