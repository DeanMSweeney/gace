import os
import lmdb
import pickle
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .lmdb_dataset import LazyLmdbDataset
from .utils import pbc_aware_disp
from ase.geometry import find_mic

def precompute_vel(
    src_path: str = "/workspace/datasets/mini/data.lmdb",
    dst_pat: str = "/workspace/datasets/mini/data_refined.lmdb",
    use_ase: bool = True, 
    map_size_mult: float = 1.5, 
    commit_every: int = 5000,
):
    
    dict = {"src": src_path}
    dataset = LazyLmdbDataset(dict)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shiffle=False,
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
    
    def read_src_item_bytes(idx: int):
        key = str(idx).encode("ascii")
        with src_env.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Missing key {idx} in source LMDB")
        return buf
    
    count = 0 
    txn = dst_env.begin(write=True)
    for idx, data in enumerate(loader):
        
        x0 = data.pos.numpy() 
        x1 = data.pos_relaxed.numpy()
        C = data.cell.numpy()
        
        if use_ase:
             # compute velocity field under minimum image convention 
             u = x1 - x0 
             u_mic, _ = find_mic(v=u, cell=C, pbc=True).astype(np.float32, copy=False)
             
        u_mic = torch.tensor(u_mic, dtype=float)
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