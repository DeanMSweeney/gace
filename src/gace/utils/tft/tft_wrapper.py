from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


@dataclass
class TensorDictBatch:
    """Container for dense transformer input and sparse<->dense bookkeeping."""

    data: TensorDict
    mask: Tensor        # (B, max_nodes) bool — True = real token
    batch_index: Tensor

    def to_sparse(self, dense: Tensor) -> Tensor:
        """Convert a dense padded tensor back to sparse using the real-token mask.

        Args:
            dense: (B, max_nodes, D) — padded tensor from transformer output
        Returns:
            sparse: (N, D) — real tokens only, preserving original atom ordering
        """
        return dense[self.mask]


class PyGToTensorDict:
    """Convert interpolated PyG `Data/Batch` objects into dense `TensorDict` inputs.

    The output schema mirrors what the copied TFT/TransformerModule stack expects.
    Primitive-cell atoms (tags==1) and adsorbate atoms (tags==2) are split into
    **separate** dense sequences so the transformer can embed/attend them independently.
    """

    def __init__(
        self,
        num_atom_types: int = 100,
    ) -> None:
        self.num_atom_types = num_atom_types

    # ── public entry point ───────────────────────────────────────────────────

    def to_tensordict(
        self,
        data: Data,
        max_prim_nodes: int=250,
        max_ads_nodes: int=15,
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

        # ── build time dicts — all components share the same per-graph t ─────
        t_prim = TensorDictBatch(
            data=TensorDict({"t": data.t}, batch_size=[batch_size], device=device),
            mask=x_prim.mask,
            batch_index=prim_batch,
        )
        t_ads = TensorDictBatch(
            data=TensorDict({"t": data.t}, batch_size=[batch_size], device=device),
            mask=x_ads.mask,
            batch_index=ads_batch,
        )

        # ── build feats dicts ────────────────────────────────────────────────
        f_prim = self._build_feats_dict(
            data=data,
            mask=x_prim.mask,
            batch_index=prim_batch,
            structure_dict=x_prim.data,
            batch_size=batch_size,
            device=device,
        )

        f_ads = self._build_feats_dict(
            data=data,
            mask=x_ads.mask,
            batch_index=ads_batch,
            structure_dict=x_ads.data,
            batch_size=batch_size,
            device=device,
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
            is_adsorbate: If True, store ads_com (graph-level) and
                ads_rel_pos (per-atom) instead of raw pos.  If False (prim),
                include frac_coords and graph-level lattice parameters.
        """
        sd = TensorDict({}, batch_size=[batch_size], device=device)

        if is_adsorbate:
            # Node-level: absolute Cartesian positions
            sd["ads_pos"], mask = to_dense_batch(
                data.ads_pos,
                atom_batch,
                max_num_nodes=max_num_nodes,
            )  # (B, N_ads, 3)

            sd["ads_atomic_numbers"], _ = to_dense_batch(
                data.ads_atomic_numbers,
                atom_batch,
                max_num_nodes=max_num_nodes,
            )  # (B, N_ads)

            sd["ads_ref_centered_pos"], _ = to_dense_batch(
                data.ads_ref_centered_pos,
                atom_batch,
                max_num_nodes=max_num_nodes,
            )  # (B, N_ads, 3)
        else:
            # Node-level: positions (already prim-only after _build_positions_path)
            sd["frac_coords"], _  = to_dense_batch(
                data.pos,
                atom_batch,
                max_num_nodes=max_num_nodes,
            ) # B, N, 3

            # Node-level: atomic numbers (long indices, mask token = NUM_CLASSES-1 = 99)
            sd["atomic_numbers"], mask = to_dense_batch(
                data.atomic_numbers,
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

    def _build_feats_dict(
        self,
        data: Data,
        mask: Tensor,
        batch_index: Tensor,
        structure_dict: TensorDict,
        batch_size: int,
        device: torch.device,
    ) -> TensorDictBatch:
        """Build a dense feats TensorDict for one group of atoms."""
        fd = TensorDict({}, batch_size=[batch_size], device=device)

        fd["ads_binding_energy"] = data.ads_binding_energy.float()  # (B,)

        return TensorDictBatch(data=fd, mask=mask, batch_index=batch_index)
