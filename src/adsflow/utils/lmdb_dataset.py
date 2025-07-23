from __future__ import annotations

import bisect
import logging
import pickle
from typing import TYPE_CHECKING, TypeVar

import lmdb
import numpy as np
import torch
from torch_geometric.data import Batch

from fairchem.core.common.registry import registry
from fairchem.core.common.typing import assert_is_instance
from fairchem.core.common.utils import pyg2_data_transform
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms

if TYPE_CHECKING:
    from pathlib import Path
    from torch_geometric.data.data import BaseData

T_co = TypeVar("T_co", covariant=True)

@registry.register_dataset("lmdb_lazy")
class LazyLmdbDataset(BaseDataset):
    """
    LMDB loader that lazily opens its env(s) in each worker, avoiding handle
    inheritance and segmentation faults when num_workers > 0.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        assert not self.config.get(
            "train_on_oc20_total_energies", False
        ), "For training on total energies set dataset=oc22_lmdb"

        assert len(self.paths) == 1, f"{type(self)} only supports one src path"
        self.path = self.paths[0]

        # _keys and num_samples from parent but do not open any env
        if not self.path.is_file():
            # discover lmdb files and count entries, WITHOUT keeping envs open
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert db_paths, f"No LMDBs in {self.path}"
            self._keys = []
            lengths = []
            for p in db_paths:
                # open temporary, get count, then close
                env = lmdb.open(str(p), readonly=True, lock=False, readahead=True, meminit=False)
                length_entry = env.begin().get(b"length")
                cnt = pickle.loads(length_entry) if length_entry else env.stat()["entries"]
                env.close()
                self._keys.append(list(range(cnt)))
                lengths.append(cnt)
            self._keylen_cumulative = np.cumsum(lengths).tolist()
            self.num_samples = sum(lengths)
        else:
            # single file
            self._keys = list(range(int(self.path.open('rb').read().count(b''))))
            self.num_samples = len(self._keys)

        # sharding metadata
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            indices = np.array_split(range(self.num_samples), self.config["total_shards"]);
            self.indices = indices[self.config.get("shard", 0)]
            self.num_samples = len(self.indices)
        else:
            self.indices = list(range(self.num_samples))

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))

        # placeholders for env(s)
        self.envs: list[lmdb.Environment] | None = None
        self.env: lmdb.Environment | None = None
        self._opened = False

    def _ensure_open(self):
        if self._opened:
            return
        if not self.path.is_file():
            self.envs = []
            for p in sorted(self.path.glob("*.lmdb")):
                self.envs.append(
                    lmdb.open(str(p), readonly=True, lock=False, readahead=True, meminit=False)
                )
        else:
            self.env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=True, meminit=False)
        self._opened = True

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> T_co:
        real_idx = self.indices[idx]
        # open LMDB handles in this worker on first access
        self._ensure_open()

        if not self.path.is_file():
            db_idx = bisect.bisect(self._keylen_cumulative, real_idx)
            el_idx = real_idx - (self._keylen_cumulative[db_idx-1] if db_idx>0 else 0)
            pickled = self.envs[db_idx].begin().get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        else:
            pickled = self.env.begin().get(f"{self._keys[real_idx]}".encode("ascii"))

        data_obj = pyg2_data_transform(pickle.loads(pickled))
        data_obj = self.transforms(data_obj)
        if self.key_mapping is not None:
            from fairchem.core.datasets._utils import rename_data_object_keys
            data_obj = rename_data_object_keys(data_obj, self.key_mapping)
        return data_obj

    def __del__(self):
        # close env(s)
        if self._opened:
            if self.envs:
                for e in self.envs:
                    e.close()
            if self.env:
                self.env.close()

# Usage in your training script:
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
#
# dataset = LazyLmdbDataset(cfg)
# loader = DataLoader(dataset, batch_size=..., num_workers=8, pin_memory=True, multiprocessing_context='spawn')
