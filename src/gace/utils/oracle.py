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
from torch_scatter import scatter

from gace.utils.tft.tft_wrapper import PyGToTensorDict
from gace.utils.builder import lattice_params_to_matrix


class Oracle:
    """Mixin providing the energy-regression oracle loss for LightFlowMatch.

    Expects the host class to define:
    - ``self.oracle_model``: the ``TransformerModuleEnergy`` module.
    - ``self.hparams.oracle_softmax_temperature``
    - ``self.hparams.oracle_guidance_min_t``
    """

    def prep_data_for_oracle(self, data):
        """Populate clean-structure conditioning features on the raw batch in place.

        - ``data.t``: per-graph conditioning time, all ones (clean structure).
        - ``data.ads_ref_centered_pos``: the reference adsorbate geometry, centered
          per graph (forge's inference conditioning, path.py:500-504). Centered in
          both train and eval — deterministic, and the forge trunk is orientation-
          robust from its own training-time rotation augmentation.
        """
        B = data.num_graphs
        data.t = torch.ones(B, device=data.pos.device)
        
        # Center adsorbate 
        ads_batch = data.batch[data.tags == 2]
        ref = data.ref_ads_pos.float()
        com = scatter(ref, ads_batch, dim=0, dim_size=B, reduce="mean")
        data.ads_ref_centered_pos = ref - com[ads_batch]  # (N_ads, 3)

        # Shift atomic numbers 
        data.atomic_numbers = data.atomic_numbers - 1
        data.ads_atomic_numbers = data.ads_atomic_numbers - 1

        # Add supercell lattice 
        n_scale = data.prim_natoms.float().unsqueeze(1) ** (1 / 3)  # (B, 1)
        C = lattice_params_to_matrix(lengths=data.lengths_scaled.squeeze(1) * n_scale,
                                     angles=data.angles_radians.squeeze(1))  # (B, 3, 3)
        data.supercell_lattice = torch.bmm(data.transformation_matrix.float(), C)  # (B, 3, 3)

        return data

    def energy_criterion(self, data, pred_prim=None, pred_ads=None, x_ads=None,
                         prim_padding_mask=None, ads_padding_mask=None) -> torch.Tensor:
        """Standalone oracle energy MAE loss on the ground-truth structure.

        build_data_feats populates data.t and data.ads_ref_centered_pos; the wrapper
        densifies everything. The two fields the wrapper can't produce — 0-indexed
        atom types and the derived supercell lattice (L_super = M @ C) — are written
        onto the tensordict here so the model call is a straight pass-through.

        Returns:
            The scalar energy MAE (standardized target).
        """
        self.prep_data_for_oracle(data)
        x_prim, x_ads, t_prim, t_ads, f_prim, f_ads = PyGToTensorDict().to_tensordict(data=data)
        t = t_prim.data["t"].squeeze(-1)  # (B,)

        energy_pred = self.oracle_model(
            x_prim=(
                x_prim.data["atomic_numbers"],          # (B, M) long, 0-indexed
                x_prim.data["frac_coords"],             # (B, M, 3)
                x_prim.data["lengths_scaled"],          # (B, 1, 3)
                x_prim.data["angles_radians"],          # (B, 1, 3)
                x_prim.data["transformation_matrix"].float(),   # (B, 3, 3)
                x_prim.data["supercell_lattice"],       # (B, 3, 3)
            ),
            x_ads=(
                x_ads.data["ads_atomic_numbers"],       # (B, N) long, 0-indexed
                x_ads.data["ads_pos"],                  # (B, N, 3)
                x_ads.data["ads_ref_centered_pos"],     # (B, N, 3)
            ),
            t=t,
            prim_padding_mask=~x_prim.mask,
            ads_padding_mask=~x_ads.mask,
        )  # (B,)

        _AVG_BINDING_ENERGY = torch.tensor(self.avg_binding_energy, device=energy_pred.device)
        _STD_BINDING_ENERGY = torch.tensor(self.std_binding_energy, device=energy_pred.device)

        target = f_prim.data["ads_binding_energy"].float()
        target = (target - _AVG_BINDING_ENERGY.float()) / _STD_BINDING_ENERGY.float()  # standardize
        return (energy_pred - target).abs().mean()
