from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


@dataclass
class TensorDictBatch:
    """Container for dense transformer input and sparse<->dense bookkeeping."""

    data: TensorDict
    mask: Tensor
    batch_index: Tensor


class PyGToTensorDict:
    """Convert interpolated PyG `Data/Batch` objects into dense `TensorDict` inputs.

    The output schema mirrors what the copied TFT/TransformerModule stack expects.
    Primitive-cell atoms (tags==1) and adsorbate atoms (tags==2) are split into
    **separate** dense sequences so the transformer can embed/attend them independently.
    """

    def __init__(
        self,
        num_atom_types: int = 100,
        default_dataset_idx: int = 1,
        default_spacegroup: int = 0,
        default_charge: float = 0.0,
        default_spin: int = 0,
    ) -> None:
        self.num_atom_types = num_atom_types
        self.default_dataset_idx = default_dataset_idx
        self.default_spacegroup = default_spacegroup
        self.default_charge = default_charge
        self.default_spin = default_spin

    # ── public entry point ───────────────────────────────────────────────────

    def to_tensordict(
        self,
        data: Data,
        max_prim_nodes: Optional[int] = None,
        max_ads_nodes: Optional[int] = None,
    ) -> Tuple[
        TensorDictBatch, TensorDictBatch,   # x_prim, x_ads
        TensorDictBatch, TensorDictBatch,   # t_prim, t_ads
        TensorDictBatch, TensorDictBatch,   # f_prim, f_ads
    ]:
        """Convert a sparse PyG batch to separate dense padded `TensorDict`s
        for primitive-cell and adsorbate atoms.

        Args:
            data: Interpolated PyG `Data` or batched `Batch`.  Must carry
                ``tags`` (1 = slab, 2 = adsorbate) and ``batch`` attributes.
            max_prim_nodes: Optional fixed token length for prim atoms.
            max_ads_nodes: Optional fixed token length for ads atoms.

        Returns:
            Six `TensorDictBatch` objects:
            ``(x_prim, x_ads, t_prim, t_ads, f_prim, f_ads)``
        """
        batch_index = data.batch.long()
        device = batch_index.device
        batch_size = int(batch_index.max().item()) + 1  # number of graphs

        # ── split atoms by tag ───────────────────────────────────────────────
        prim_mask = (data.tags == 1)  # (N_total,) bool
        ads_mask = (data.tags == 2)   # (N_total,) bool

        prim_batch = batch_index[prim_mask]
        ads_batch = batch_index[ads_mask]

        if max_prim_nodes is None:
            max_prim_nodes = int(torch.bincount(prim_batch, minlength=batch_size).max().item())
        if max_ads_nodes is None:
            max_ads_nodes = int(torch.bincount(ads_batch, minlength=batch_size).max().item())

        if max_prim_nodes <= 0:
            raise ValueError("max_prim_nodes must be positive (no prim atoms found).")
        if max_ads_nodes <= 0:
            raise ValueError("max_ads_nodes must be positive (no ads atoms found).")

        # ── build structure dicts ────────────────────────────────────────────
        x_prim = self._build_structure_dict(
            data=data,
            atom_mask=prim_mask,
            atom_batch=prim_batch,
            batch_size=batch_size,
            max_num_nodes=max_prim_nodes,
            device=device,
        )

        x_ads = self._build_structure_dict(
            data=data,
            atom_mask=ads_mask,
            atom_batch=ads_batch,
            batch_size=batch_size,
            max_num_nodes=max_ads_nodes,
            device=device,
            is_adsorbate=True,
        )

        # ── build time dicts ─────────────────────────────────────────────────
        t_prim = self._build_time_dict(
            data=data,
            atom_mask=prim_mask,
            atom_batch=prim_batch,
            dense_mask=x_prim.mask,
            batch_size=batch_size,
            max_num_nodes=max_prim_nodes,
            device=device,
        )

        t_ads = self._build_time_dict(
            data=data,
            atom_mask=ads_mask,
            atom_batch=ads_batch,
            dense_mask=x_ads.mask,
            batch_size=batch_size,
            max_num_nodes=max_ads_nodes,
            device=device,
            is_adsorbate=True,
        )

        # ── build feats dicts ────────────────────────────────────────────────
        f_prim = self._build_feats_dict(
            mask=x_prim.mask,
            batch_index=prim_batch,
            structure_dict=x_prim.data,
            batch_size=batch_size,
            device=device,
            is_periodic=True,
        )

        f_ads = self._build_feats_dict(
            mask=x_ads.mask,
            batch_index=ads_batch,
            structure_dict=x_ads.data,
            batch_size=batch_size,
            device=device,
            is_periodic=False,
        )

        return x_prim, x_ads, t_prim, t_ads, f_prim, f_ads

    # ── private builders ─────────────────────────────────────────────────────

    def _build_structure_dict(
        self,
        data: Data,
        atom_mask: Tensor,
        atom_batch: Tensor,
        batch_size: int,
        max_num_nodes: int,
        device: torch.device,
        is_adsorbate: bool = False,
    ) -> TensorDictBatch:
        """Build a dense structure TensorDict for one group of atoms.

        Args:
            atom_mask: Boolean mask selecting atoms from the combined flat tensors.
            atom_batch: Per-atom graph assignment for the selected atoms.
            is_adsorbate: If True, store ads_center (graph-level) and
                ads_rel_pos (per-atom) instead of raw pos.  If False (prim),
                include frac_coords and graph-level lattice parameters.
        """
        sd = TensorDict({}, batch_size=[batch_size], device=device)

        # Node-level: atomic_numbers_hot
        sd["atomic_numbers_hot"], mask = to_dense_batch(
            data.atomic_numbers_hot[atom_mask],
            atom_batch,
            max_num_nodes=max_num_nodes,
        )

        if is_adsorbate:
            # Graph-level: adsorbate center of mass
            sd["ads_center"] = data.ads_center.unsqueeze(1)  # (B, 3) → (B, 1, 3)

            # Node-level: positions relative to COM
            sd["ads_rel_pos"], _ = to_dense_batch(
                data.ads_rel_pos[atom_mask],
                atom_batch,
                max_num_nodes=max_num_nodes,
            )
        else:
            # Node-level: fractional coordinates
            sd["frac_coords"], _ = to_dense_batch(
                data.pos[atom_mask],
                atom_batch,
                max_num_nodes=max_num_nodes,
            )

            # Graph-level lattice parameters (prim only)
            if hasattr(data, "lengths_scaled"):
                sd["lengths_scaled"] = data.lengths_scaled.unsqueeze(1)  # (B, 1, 3)
            if hasattr(data, "angles_radians"):
                sd["angles_radians"] = data.angles_radians.unsqueeze(1)  # (B, 1, 3)
            if hasattr(data, "transformation_matrix"):
                sd["transformation_matrix"] = data.transformation_matrix  # (B, 3, 3)
            if hasattr(data, "supercell_lattice"):
                sd["supercell_lattice"] = data.supercell_lattice  # (B, 3, 3) = S @ L

        return TensorDictBatch(data=sd, mask=mask, batch_index=atom_batch)

    def _build_time_dict(
        self,
        data: Data,
        atom_mask: Tensor,
        atom_batch: Tensor,
        dense_mask: Tensor,
        batch_size: int,
        max_num_nodes: int,
        device: torch.device,
        is_adsorbate: bool = False,
    ) -> TensorDictBatch:
        """Build a dense time TensorDict for one group of atoms."""
        td = TensorDict({}, batch_size=[batch_size], device=device)

        # Node-level times
        for key in ["t_x", "t_h"]:
            if hasattr(data, key):
                td[key], _ = to_dense_batch(
                    getattr(data, key)[atom_mask],
                    atom_batch,
                    max_num_nodes=max_num_nodes,
                )

        if is_adsorbate:
            # Graph-level ads center time
            if hasattr(data, "t_ads_center"):
                td["t_ads_center"] = data.t_ads_center.unsqueeze(1)  # (B, 1)
        else:
            # Graph-level lattice times (prim only)
            for key in ["t_Lengths", "t_angles", "t_transformation_matrix"]:
                if hasattr(data, key):
                    td[key] = getattr(data, key).unsqueeze(1)  # (B, 1, …)

        return TensorDictBatch(data=td, mask=dense_mask, batch_index=atom_batch)

    def _build_feats_dict(
        self,
        mask: Tensor,
        batch_index: Tensor,
        structure_dict: TensorDict,
        batch_size: int,
        device: torch.device,
        is_periodic: bool,
    ) -> TensorDictBatch:
        """Build a dense feats TensorDict for one group of atoms."""
        fd = TensorDict({}, batch_size=[batch_size], device=device)

        num_tokens = structure_dict["atomic_numbers_hot"].size(1)  # padded seq len
        atom_to_token_idx = torch.arange(num_tokens, device=device).expand(batch_size, -1)
        atom_to_token = F.one_hot(atom_to_token_idx, num_classes=num_tokens).to(torch.float32)
        max_num_tokens = mask.sum(dim=1).to(torch.long)
        padding_mask = ~mask

        fd["token_is_periodic"] = torch.full(
            (batch_size, num_tokens), is_periodic, device=device, dtype=torch.bool,
        )
        fd["dataset_idx"] = torch.full(
            (batch_size,), self.default_dataset_idx, device=device, dtype=torch.long,
        )
        fd["spacegroup"] = torch.full(
            (batch_size,), self.default_spacegroup, device=device, dtype=torch.long,
        )
        fd["charge"] = torch.full(
            (batch_size,), self.default_charge, device=device, dtype=torch.float32,
        )
        fd["spin"] = torch.full(
            (batch_size,), self.default_spin, device=device, dtype=torch.long,
        )
        fd["ref_pos"] = torch.zeros_like(
            structure_dict["atom_pos"], dtype=torch.float32,
        )
        fd["ref_space_uid"] = atom_to_token_idx
        fd["atom_to_token"] = atom_to_token
        fd["atom_to_token_idx"] = atom_to_token_idx
        fd["token_index"] = atom_to_token_idx
        fd["max_num_tokens"] = max_num_tokens
        fd["padding_mask"] = padding_mask

        return TensorDictBatch(data=fd, mask=mask, batch_index=batch_index)
