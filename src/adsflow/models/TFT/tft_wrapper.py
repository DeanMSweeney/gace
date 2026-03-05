from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    """

    def __init__(
        self,
        num_atom_types: int = 100,
        default_dataset_idx: int = 1,
        default_spacegroup: int = 0,
        default_charge: float = 0.0,
        default_spin: int = 0,
        default_token_is_periodic: bool = False,
    ) -> None:
        self.num_atom_types = num_atom_types
        self.default_dataset_idx = default_dataset_idx
        self.default_spacegroup = default_spacegroup
        self.default_charge = default_charge
        self.default_spin = default_spin
        self.default_token_is_periodic = default_token_is_periodic

    def to_tensordict(
        self,
        data: Data,
        max_num_nodes: Optional[int] = None,
    ) -> TensorDictBatch:
        """Convert a sparse PyG batch to dense padded `TensorDict`.

        Args:
            data: Interpolated PyG `Data` or batched `Batch`.
            max_num_nodes: Optional fixed token length. If omitted, use batch max atoms.

        Returns:
            `TensorDictBatch` with:
            - `data`: dense TensorDict for transformer forward
            - `mask`: real-token mask (`True` for non-padding)
            - `batch_index`: sparse->graph assignment (`N_total,`)
        """
        batch_index = data.batch.long()
        num_nodes = batch_index.numel()
        device = batch_index.device

        if max_num_nodes is None:
            max_num_nodes = int(torch.bincount(batch_index).max().item())
        if max_num_nodes <= 0:
            raise ValueError("max_num_nodes must be positive.")

        batch_size = int(batch_index.max().item()) + 1
        dtype = data.pos.dtype if hasattr(data, "pos") else torch.float32

        # extract per node elements
        structure_dict = TensorDict({}, batch_size=[batch_size], device=device)
        x_keys = ["atomic_numbers_hot", "frac_coords", "token_is_periodic", "lengths_scaled", "angles_radians"]
        for key in x_keys:
            if key == "atomic_numbers_hot":
                structure_dict[key], mask = self._get_node_graph_tensor(
                data=data, 
                key=key,
                element="node",
                dtype=torch.float32,
                batch_index=batch_index,
                max_num_nodes=max_num_nodes,
            )
            else:
                if key in ["lengths_scaled", "angles_radians"]:
                    element = "graph"
                else:
                    element = "node"
                structure_dict[key], _ = self._get_node_graph_tensor(
                    data=data, 
                    key=key,
                    element=element,
                    dtype=torch.float32,
                    batch_index=batch_index,
                    max_num_nodes=max_num_nodes,
                )
                if element == "graph":
                    structure_dict[key] = structure_dict[key].unsqueeze(1)
        structure_dict["pos"] = torch.zeros_like(structure_dict["frac_coords"], dtype=torch.float32)

        # extract per node elements
        time_dict = TensorDict({}, batch_size=[batch_size], device=device)
        time_keys = ["t_x", "t_h", "t_Lengths", "t_angles"]
        for key in time_keys:
                if key in ["t_Lengths", "t_angles"]:
                    element = "graph"
                else:
                    element = "node"
                time_dict[key], _ = self._get_node_graph_tensor(
                    data=data, 
                    key=key,
                    element=element,
                    dtype=torch.float32,
                    batch_index=batch_index,
                    max_num_nodes=max_num_nodes,
                )
                if element == "graph":
                    time_dict[key] = time_dict[key].unsqueeze(1)

        dataset_idx = torch.full((batch_size,), self.default_dataset_idx, device=device, dtype=torch.long)
        spacegroup = torch.full((batch_size,), self.default_spacegroup, device=device, dtype=torch.long)
        charge = torch.full((batch_size,), self.default_charge, device=device, dtype=torch.float32)
        spin = torch.full((batch_size,), self.default_spin, device=device, dtype=torch.long)

        num_tokens = structure_dict["atomic_numbers_hot"].size(1)
        atom_to_token_idx = torch.arange(num_tokens, device=device).expand(batch_size, -1)
        atom_to_token = F.one_hot(atom_to_token_idx, num_classes=num_tokens).to(torch.float32)
        max_num_tokens = mask.sum(dim=1).to(torch.long)
        padding_mask = ~mask # oppostie of mask 

        # miscellaneous data 
        feats_dict = TensorDict({}, batch_size=[batch_size], device=device)
        feats_dict["dataset_idx"] = dataset_idx
        feats_dict["spacegroup"] = spacegroup
        feats_dict["charge"] = charge
        feats_dict["spin"] = spin
        feats_dict["ref_pos"] = torch.zeros_like(structure_dict["frac_coords"], dtype=torch.float32)
        feats_dict["ref_space_uid"] = atom_to_token_idx
        feats_dict["atom_to_token"] = atom_to_token
        feats_dict["atom_to_token_idx"] = atom_to_token_idx
        feats_dict["token_index"] = atom_to_token_idx
        feats_dict["max_num_tokens"] = max_num_tokens
        feats_dict["padding_mask"] = padding_mask

        x = TensorDictBatch(data=structure_dict, mask=mask, batch_index=batch_index)
        t = TensorDictBatch(data=time_dict, mask=mask, batch_index=batch_index)
        f = TensorDictBatch(data=feats_dict, mask=mask, batch_index=batch_index)

        return x, t, f

    @staticmethod
    def _get_node_graph_tensor(
        data: Data,
        key: str,
        element: str,
        dtype: torch.dtype,
        batch_index: Tensor,
        max_num_nodes: int,
    ) -> Tensor:
        if not hasattr(data, key):
            raise ValueError(f"Could not build `{key}`. Missing `{key}`.")
        value = getattr(data, key)

        if element == "node":
            return to_dense_batch(value, batch_index, max_num_nodes=max_num_nodes)
        elif element == "graph":
            return value, None
        else:
            raise ValueError(f"Unsupported element type: {element}")
