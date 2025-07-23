from __future__ import annotations

import bisect
import pickle
from typing import TYPE_CHECKING, TypeVar

import lmdb
import numpy as np
from torch_geometric.data.data import BaseData

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import pyg2_data_transform
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms

if TYPE_CHECKING:
    from pathlib import Path

T_co = TypeVar("T_co", covariant=True)

@registry.register_dataset("lmdb_lazy")
class LazyLmdbDataset(BaseDataset):
    """
    LMDB loader that lazily opens its environments in each worker,
    avoiding handle inheritance and segmentation faults when num_workers > 0.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        assert len(self.paths) == 1, f"{type(self)} only supports one src path"
        self.path = self.paths[0]

        # Discover LMDB shards (or single file) and count entries without keeping env open
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
        else:
            db_paths = [self.path]

        self._keys = []
        lengths = []
        for p in db_paths:
            env = lmdb.open(str(p), readonly=True, lock=False, readahead=True, meminit=False)
            length_entry = env.begin().get(b"length")
            cnt = pickle.loads(length_entry) if length_entry else env.stat().get("entries", 0)
            env.close()
            self._keys.append(list(range(cnt)))
            lengths.append(cnt)

        self._keylen_cumulative = np.cumsum(lengths).tolist()
        self.num_samples = sum(lengths)

        # Sharding support
        if "shard" in self.config and "total_shards" in self.config:
            all_indices = list(range(self.num_samples))
            shards = np.array_split(all_indices, self.config["total_shards"])
            self.indices = shards[self.config.get("shard", 0)].tolist()
        else:
            self.indices = list(range(self.num_samples))

        self.transforms = DataTransforms(self.config.get("transforms", {}))
        self.key_mapping = self.config.get("key_mapping", None)

        # Placeholder for per-process LMDB environments
        self.envs: list[lmdb.Environment] | None = None
        self._opened = False

    def _ensure_open(self):
        if self._opened:
            return
        self.envs = []
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
        else:
            db_paths = [self.path]
        for p in db_paths:
            env = lmdb.open(str(p), readonly=True, lock=False, readahead=True, meminit=False)
            self.envs.append(env)
        self._opened = True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> BaseData:
        real_idx = self.indices[idx]
        self._ensure_open()

        # Determine which shard and element index
        db_idx = bisect.bisect(self._keylen_cumulative, real_idx)
        rel_idx = real_idx - (self._keylen_cumulative[db_idx - 1] if db_idx > 0 else 0)
        pickled = self.envs[db_idx].begin().get(f"{self._keys[db_idx][rel_idx]}".encode("ascii"))

        data = pyg2_data_transform(pickle.loads(pickled))
        data = self.transforms(data)
        if self.key_mapping:
            from fairchem.core.datasets._utils import rename_data_object_keys
            data = rename_data_object_keys(data, self.key_mapping)
        return data

    def __del__(self):
        if self._opened and self.envs:
            for env in self.envs:
                env.close()
