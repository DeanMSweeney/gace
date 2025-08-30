"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations  # enable postponed type annotations

import bisect                       # for binary-search on cumulative lengths
import pickle                       # for deserializing stored entries
from typing import TYPE_CHECKING, TypeVar

import lmdb                         # LMDB environment bindings
import numpy as np                  # numeric operations
import torch                        # PyTorch core
from torch_geometric.data import Batch  # for collating Data objects

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

@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")
class LazyLmdbDataset(BaseDataset):
    """Dataset for OC20 LMDB tasks (S2EF, IS2RS, IS2RE)."""

    def __init__(self, config) -> None:
        super().__init__(config)  # sets self.paths from config
        assert len(self.paths) == 1, f"{type(self)} requires a single src path"
        self.path = self.paths[0]

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

        # deserialize and convert to PyG Data
        data = pyg2_data_transform(pickle.loads(raw))

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

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple


class OptTransportMixin: 
    """
    This Mixin supports image invariant interpolation paths and masked loss for frozen atoms 
    """
   
    @staticmethod # define as normal function within class's namespace
    def pbc_aware_disp(
                cell: Tensor, 
                pos: Tensor, 
                pos_relaxed: Tensor, 
                batch: Tensor, 
                naive: bool,
    ) -> Tensor:
          
          A  =  cell[batch]  # (N,3,3)
          G = A @ A.transpose(1,2) # (N, 3, 3)
          x0 = pos # (N,3)
          x1 = pos_relaxed # (N,3)
          N = len(batch)
          
          u = x1 - x0 # (N,3)
          u_f = torch.linalg.solve(A.transpose(1,2), u.unsqueeze(-1)).squeeze(-1)  # (N,3)
          n0 = torch.round(u_f)

          if naive:
                print("Naive method")
                u_f_min = u_f - n0
          else: 
                print("27 image search")
                # nearest neighbor search algorithm 
                D = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), dtype=torch.int) # (27, 3)
                n = n0.unsqueeze(1) + D.unsqueeze(0) # (N, 1, 3) + (1, 27, 3) = (N, 27, 3) 
                U_f_n = u_f.unsqueeze(1) - n # (N, 1, 3) - (N, 27, 3) = (N, 27, 3)
                W = U_f_n @ G # (N, 27, 3)
                u_f2 = (W * U_f_n).sum(dim=-1) # (N, 27)
                  
                # output 
                k_min = u_f2.argmin(dim=1) # N
                u_f_min = U_f_n[torch.arange(N), k_min]
                
          u_min = torch.bmm(u_f_min.unsqueeze(1), A).squeeze(1) 
          return u_min


### Testing ###
if __name__ == "__main__":
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import itertools
    
    dict = {"src": "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"}
    dataset = LazyLmdbDataset(dict)
    loader = DataLoader(
        dataset=dataset,
        batch_size=10000,
        pin_memory=True,
    )
    
    u1 = []
    u2 = []
    for data in loader: 
            naive = OptTransportMixin.pbc_aware_disp(
                  cell = data.cell,
                  pos = data.pos, 
                  pos_relaxed = data.pos_relaxed, 
                  batch = data.batch,
                  naive=True,
            )
            
            neighbor = naive = OptTransportMixin.pbc_aware_disp(
                  cell = data.cell,
                  pos = data.pos, 
                  pos_relaxed = data.pos_relaxed, 
                  batch = data.batch,
                  naive=False
            )
            
            u1.append(naive)
            u2.append(neighbor)
            
    print(u1 == u2)
        
            
     
            
            
            
            
            
    
                
