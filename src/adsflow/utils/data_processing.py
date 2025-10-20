import os
import lmdb
import pickle
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from adsflow.utils.lmdb_dataset import LazyLmdbDataset
from adsflow.utils.optimal_transport import OptTransportMixin as OTM


def precompute_vel(
    src_path: str = "/workspace/datasets/mini/data.lmdb",
    dst_path: str = "/workspace/datasets/mini/data_refined.lmdb",
    map_size_mult: float = 1.5, 
    commit_every: int = 10,
    batch_size: int = 1,
    num_workers: int = 0,
):
    
    cfg = {"src": src_path}
    dataset = LazyLmdbDataset(cfg)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    src_size = os.path.getsize(src_path)
    map_size = int(src_size * map_size_mult)
    
    src_env = lmdb.open(
        src_path,
        subdir=False,
        readonly=True, 
        lock=False,
        readahead=True, 
        meminit=False,
    )
    
    dst_env = lmdb.open(
        dst_path,
        subdir=False,
        map_size=map_size,
        lock=True,
        readahead=False, 
        meminit=False,
    )
        
    count = 0 
    txn = dst_env.begin(write=True)
    
    print(">>> Writing velocity to new data file...")
    for idx, data in enumerate(loader):
       
        _, velocity = OTM.path(
            input_data=data,
            from_noise=False,
            otf_velocity=True,
            naive=True,
        )
        
        data.velocity = velocity.detach().cpu()
        
        key = str(idx).encode("ascii")
        txn.put(key, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        count += 1
        
        if count % commit_every == 0:
            txn.commit()
            txn = dst_env.begin(write=True)
            print(f">>> Committed {count} samples")

    # Final commit + close
    txn.commit()
    src_env.close()
    dst_env.close()

    print(f"Wrote {count} samples with 'velocity' to: {dst_path}")

if __name__ == "__main__":
    import time
    
    start = time.time()
    precompute_vel(
        src_path="/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb",
        dst_path="/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data_refined.lmdb",
        batch_size=1,
        commit_every=5000,
    )
    end = time.time()
    
    print(f"Elapsed time: {end - start}")

    

