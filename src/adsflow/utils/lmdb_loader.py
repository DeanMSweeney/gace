"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# =============================================================================
# Edited by Dean M. Sweeney on 07/25
# =============================================================================
# 1. Added **lazy LMDB opening** (envs opened on first access, not at init).
# 2. Disabled **LMDB readahead** to avoid huge I/O prefetch on HPC filesystems.
# 3. Computed shard lengths using brief open/close instead of persistent envs.
# 4. Simplified global→local index mapping using `bisect_right`.
# 5. Renamed class to `LazyLmdbDataset` for clarity; core data decoding unchanged.
#
# All OC20 behavior preserved; design now avoids heavy startup cost and reduces memory/I/O.
# =============================================================================

from __future__ import annotations  # enable postponed type annotations

import bisect                       # for binary-search on cumulative lengths
import pickle                       # for deserializing stored entries
from typing import TYPE_CHECKING, TypeVar

import logging 
import lmdb                         # LMDB environment bindings
import numpy as np                  # numeric operations
import torch                        # PyTorch core
from torch_geometric.data import Batch, Data  # for collating Data objects

from fairchem.core.common.registry import registry
from fairchem.core.common.typing import assert_is_instance
from fairchem.core.common.utils import pyg2_data_transform
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms

if TYPE_CHECKING:
    from pathlib import Path
    from torch_geometric.data.data import BaseData

T_co = TypeVar("T_co", covariant=True)

def dict_to_pyg_data(data_dict: dict) -> Data:
    """
    Convert a dictionary to a PyG Data object.

    Core fields (pos, cell, atomic_numbers, natoms, fixed, tags, sid, fid) are
    always required. Optional fields are added only if present in the dict,
    so both new (raw-structure-only) and old (OC20-dense rich) LMDBs load cleanly.

    Args:
        data_dict: Dictionary with LMDB fields

    Returns:
        PyG Data object with all available fields as torch tensors
    """
    # --- Required core fields ---
    kwargs = dict(
        pos=torch.from_numpy(data_dict["pos"]),
        cell=torch.from_numpy(data_dict["cell"]).unsqueeze(0),
        atomic_numbers=torch.from_numpy(data_dict["atomic_numbers"]),
        natoms=torch.tensor(data_dict["natoms"]),
        fixed=torch.from_numpy(np.array(data_dict["fixed"])),
        tags=torch.from_numpy(np.array(data_dict["tags"])),
        sid=data_dict["sid"],
        fid=torch.tensor(data_dict["fid"]),
    )

    # --- Optional fields: added only when present in the stored dict ---
    _optional_numpy = ["pos_relaxed", "velocity"]
    _optional_scalar = ["Fmax", "dE", "dSmax", "y_init", "y_relaxed"]

    for field in _optional_numpy:
        if field in data_dict:
            kwargs[field] = torch.from_numpy(np.array(data_dict[field]))

    for field in _optional_scalar:
        if field in data_dict:
            kwargs[field] = torch.tensor(data_dict[field])

    return Data(**kwargs)

@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")

class LazyLmdbDataset(BaseDataset):
    """Dataset for OC20 LMDB tasks (S2EF, IS2RS, IS2RE)."""

    def __init__(self, config) -> None:
        super().__init__(config)  # sets self.paths from config
        assert len(self.paths) == 1, f"{type(self)} requires a single src path"
        self.path = self.paths[0]
        print(self.path)

        # choose one or multiple LMDB files
        if self.path.is_file():
            db_paths = [self.path]
        else:
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert db_paths, f"No LMDB files in '{self.path}'"

        # store paths; will lazy-open environments
        self.db_paths = db_paths
        self.envs = [None] * len(db_paths)
        self._keys = []  # per-shard key counts

        # compute number of entries per shard without holding env open
        total = 0
        for p in db_paths:
            # open briefly for stats, with readahead disabled to avoid massive prefetch
            env = lmdb.open(
                str(p), subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=1
            )
            # read stored length key or fallback to stat
            length_entry = env.begin().get(b"length")
            if length_entry is not None:
                num = pickle.loads(length_entry)
            else:
                num = assert_is_instance(env.stat()["entries"], int)
            env.close()

            self._keys.append(list(range(num)))
            total += num

        # build cumulative lengths for binary search
        lengths = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(lengths).tolist()
        self.num_samples = total

        # software sharding (optional)
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            idxs = range(self.num_samples)
            shards = np.array_split(idxs, self.config["total_shards"])
            self.indices = shards[self.config["shard"]]
            self.num_samples = len(self.indices)

        # transforms & key renaming
        self.key_mapping = self.config.get("key_mapping")
        self.transforms = DataTransforms(self.config.get("transforms", {}))

    def connect_db(self, path: Path) -> lmdb.Environment:
        """
        Open an LMDB environment with readahead disabled to prevent heavy prefetch.
        """
        return lmdb.open(
            str(path), subdir=False, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=1
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> T_co:
        # apply software sharding if enabled
        if self.sharded:
            idx = self.indices[idx]

        # map global idx to shard index and local position
        shard_idx = bisect.bisect_right(self._keylen_cumulative, idx)
        prev = self._keylen_cumulative[shard_idx - 1] if shard_idx else 0
        local_idx = idx - prev
        assert local_idx >= 0

        # lazy-open the environment for this shard if needed
        if self.envs[shard_idx] is None:
            self.envs[shard_idx] = self.connect_db(self.db_paths[shard_idx])

        env = self.envs[shard_idx]
        key = f"{self._keys[shard_idx][local_idx]}".encode("ascii")
        raw = env.begin().get(key)  # zero-copy retrieval

        # deserialize
        loaded_data = pickle.loads(raw)

        # check if it's a dictionary (from lmdb_builder) and convert to PyG Data
        if isinstance(loaded_data, dict):
            data = dict_to_pyg_data(loaded_data)
        else:
            # already a PyG Data object, apply pyg2 transform
            data = pyg2_data_transform(loaded_data)

        # user-postprocessing
        data = self.transforms(data)
        if self.key_mapping:
            data = rename_data_object_keys(data, self.key_mapping)

        return data

    def __del__(self):
        # close any opened environments
        for env in self.envs:
            if env is not None:
                env.close()

def data_list_collater(
    data_list: list[BaseData], otf_graph: bool = False, to_dict: bool = False
) -> BaseData | dict[str, torch.Tensor]:
    """Batch a list of PyG Data objects into one Batch."""
    batch = Batch.from_data_list(data_list)
    if not otf_graph:
        try:
            neighbors = [d.edge_index.shape[1] for d in data_list]
            batch.neighbors = torch.tensor(neighbors)
        except Exception:
            logging.warning("Missing edge_index; set otf_graph=True")
    return dict(batch.items()) if to_dict else batch


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    # build trianing set
    t = "/datasets/oc20/full/is2s_train_val_test_lmdbs/train"
    p = {"src": t}
    dataset = LazyLmdbDataset(p)

    train_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
        )

    # Collect all dE, Fmax, and dS values from the dataset
    dE = []
    Fmax = []
    dS = []

    print("Collecting dE, Fmax, and dS values from dataset...")
    for i, data in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Processing batch {i}/{len(train_loader)}")

        # Collect values from batch
        dE.append(data.dE)
        Fmax.append(data.Fmax)
        dS.append(data.dSmax)

        print(data.dE)

    # Concatenate lists into single tensors
    dE_tensor = torch.cat(dE, dim=0)
    Fmax_tensor = torch.cat(Fmax, dim=0)
    dS_tensor = torch.cat(dS, dim=0)

    # Save tensors to disk
    print(f"\nSaving tensors with {len(dE_tensor)} samples...")
    torch.save(dE_tensor, "dE_values.pt")
    torch.save(Fmax_tensor, "Fmax_values.pt")
    torch.save(dS_tensor, "dS_values.pt")

    print(f"Saved tensors:")
    print(f"  - dE_values.pt: shape={dE_tensor.shape}, mean={dE_tensor.mean():.4f}, std={dE_tensor.std():.4f}")
    print(f"  - Fmax_values.pt: shape={Fmax_tensor.shape}, mean={Fmax_tensor.mean():.4f}, std={Fmax_tensor.std():.4f}")
    print(f"  - dS_values.pt: shape={dS_tensor.shape}, mean={dS_tensor.mean():.4f}, std={dS_tensor.std():.4f}")