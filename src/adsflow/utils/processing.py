import os
import lmdb
import pickle
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from adsflow.utils.lmdb_dataset import LazyLmdbDataset
from adsflow.utils.flow_matching import OptTransportMixin

class GenVel(OptTransportMixin):
    
    def __init__(
        self,
        src_path: str,
        dst_path: str, 
        map_size_mult: float = 1.5, 
        commit_every: int = 5000,
        batch_size: int = 400000, 
    ):
    
    def read_src_item_bytes(self, idx: int):
        key = str(idx).encode("ascii")
        with src_env.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Missing key {idx} in source LMDB")
        return buf

    def precompute_vel(
        self,
    ):
        dict = {"src": self.src_path}
        dataset = LazyLmdbDataset(dict)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        src_size = os.path.getsize(src_path)
        map_size = int(src_size * map_size_mult)
        
        src_env = lmdb.open(
            src_path,
            subdir=False,
            readonly=True, 
            lock=False,
            readahead=True, 
            meninit=False,
        )
        
        dst_env = lmdb.open(
            src_path,
            subdir=False,
            map_size=map_size,
            lock=True,
            readahead=False, 
            meninit=False,
        )
        
        count = 0 
        txn = dst_env.begin(write=True)
        for idx, data in enumerate(loader):
            
            velocity = self.path(data, otf_velocity=True, from_noise=False)
            src_btyes = read_src_item_bytes(idx)
            sample = pickle.loads(src_btyes)
            
            sample["velocity"] = u_mic
            
            key = str(idx).encode("ascii")
            txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
            count += 1
            
            if count % commit_every == 0:
                txn.commit()
                txn = dst_env.begin(write=True)

        # Final commit + close
        txn.commit()
        src_env.close()
        dst_env.close()

        print(f"Wrote {count} samples with 'velocity' to: {dst_path}")

if __name__ == "__main__":
    precompute_vel()