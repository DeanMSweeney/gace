"""Energy-regression oracle loss mixin.

The oracle (``TransformerModuleEnergy``) maps a slab+adsorbate structure
to a scalar binding energy. ``Oracle`` is a mixin for ``LightFlowMatch``
that provides ``energy_criterion``, computing the energy MAE for both of the
oracle's roles, selected by whether a flow-model prediction is supplied:

- **Standalone training** (``pred_prim is None``): inputs come from the
  ground-truth structure via the wrapper; atom types are a hard one-hot
  (== an ``nn.Embedding`` lookup); target = the data binding energy.
- **Guidance** (``pred_prim`` given): inputs come from the flow model's dense
  endpoint prediction x_hat_1; atom types are a softmax simplex so the gradient
  flows back into the flow z-head; target = the conditioning binding energy.
"""

import torch
import torch.nn.functional as F

from gace.utils.tft.tft_wrapper import PyGToTensorDict
from gace.utils.builder import lattice_params_to_matrix

_AVG_BINDING_ENERGY = -1.54790583
_STD_BINDING_ENERGY = 2.27278609


class Oracle:
    """Mixin providing the energy-regression oracle loss for LightFlowMatch.

    Expects the host class to define:
    - ``self.oracle_model``: the ``TransformerModuleEnergy`` module.
    - ``self.hparams.oracle_softmax_temperature``
    - ``self.hparams.oracle_guidance_min_t``
    """

    def energy_criterion(self, data, pred_prim=None, pred_ads=None, x_ads=None,
                         prim_padding_mask=None, ads_padding_mask=None) -> torch.Tensor:
        """Oracle energy MAE loss, shared by standalone training and guidance.

        The supercell lattice (L_super = M @ C) is a *derived* quantity not stored
        on the raw data object, so it is recomputed from lengths/angles/M in both
        cases (mirrors path.py ``_build_transformation_matrix_path``).

        Args:
            data: PyG batch. For guidance this is the interpolated build_path data;
                for standalone it is the raw ground-truth batch.
            pred_prim, pred_ads: Dense flow-model endpoint prediction (guidance only).
            x_ads: Wrapper ads batch, for the conditioning ads atom types (guidance).
            prim_padding_mask, ads_padding_mask: Padding masks (guidance only; the
                standalone branch builds them from the wrapper).

        Returns:
            The scalar energy MAE.
        """
        guidance = pred_prim is not None

        if guidance:
            # Guidance: flow model's dense endpoint prediction.
            T = float(self.hparams.oracle_softmax_temperature)
            atom_types     = torch.softmax(pred_prim[0] / T, dim=-1)  # softmax simplex
            frac_coords    = pred_prim[1]
            lengths_scaled = pred_prim[2]
            angles_radians = pred_prim[3]
            M              = pred_prim[4]
            ads_atom_types = x_ads.data["ads_atomic_numbers"]
            ads_pos        = pred_ads[0]
            target         = data.ads_binding_energy.float()
        else:
            # Standalone: ground-truth structure from the wrapper. The shared
            # wrapper builds t_prim/t_ads from data.t; the energy model ignores
            # time, so a clean-structure placeholder (t=1) just satisfies it.
            if not hasattr(data, "t") or data.t is None:
                data.t = torch.ones(data.num_graphs, device=data.pos.device)
            wrapper = PyGToTensorDict()
            x_prim, x_ads, t_prim, t_ads, f_prim, f_ads = wrapper.to_tensordict(data=data)
            prim_padding_mask = ~x_prim.mask
            ads_padding_mask  = ~x_ads.mask

            V = self.oracle_model.atom_dim
            atom_types     = F.one_hot(x_prim.data["atomic_numbers"].long(), V).float()  # hard one-hot
            frac_coords    = x_prim.data["frac_coords"]
            lengths_scaled = x_prim.data["lengths_scaled"]
            angles_radians = x_prim.data["angles_radians"]
            M              = x_prim.data["transformation_matrix"].float()
            ads_atom_types = x_ads.data["ads_atomic_numbers"]
            ads_pos        = x_ads.data["ads_pos"]
            target         = f_prim.data["ads_binding_energy"].float()

        # Supercell lattice L_super = M @ C, C built from denormalised lengths/angles.
        n_scale = data.prim_natoms.float().unsqueeze(1) ** (1 / 3)  # (B, 1)
        C = lattice_params_to_matrix(lengths=lengths_scaled.squeeze(1) * n_scale,
                                     angles=angles_radians.squeeze(1))  # (B, 3, 3)
        supercell_lattice = torch.bmm(M, C)  # (B, 3, 3)

        energy_pred = self.oracle_model(
            x_prim=(atom_types, frac_coords, lengths_scaled, angles_radians, M, supercell_lattice),
            x_ads=(ads_atom_types, ads_pos),
            prim_padding_mask=prim_padding_mask,
            ads_padding_mask=ads_padding_mask,
        )  # (B,)

        target = (target.float() - _AVG_BINDING_ENERGY.float()) / _STD_BINDING_ENERGY.float() # standardize to zero 
        err = (energy_pred - target).abs()
        # Guidance only: optionally restrict to larger t, where x_hat_1 is a more
        # reliable clean-structure estimate.
        min_t = float(self.hparams.oracle_guidance_min_t)
        if guidance and min_t > 0.0:
            gate = (data.t.reshape(-1) >= min_t).float()
            return (err * gate).sum() / gate.sum().clamp(min=1)
        return err.mean()
