import lmdb, pickle
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from fairchem.core.common.utils import pyg2_data_transform

class LMDBIterableDataset(IterableDataset):
    def __init__(self, lmdb_path: str, shuffle: bool = False):
        self.lmdb_path = lmdb_path
        self.shuffle = shuffle

    def __iter__(self):
        # 1) Figure out this worker’s ID and how many total workers there are
        worker = get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1

        # 2) Open LMDB here, in *this* worker
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        txn = env.begin()
        cursor = txn.cursor()

        # 3) Grab all the keys once
        keys = [k for k, _ in cursor]
        if self.shuffle and wid == 0:
            import random
            random.shuffle(keys)
        # (optional: broadcast a single shuffle order to all workers)

        # 4) Yield only every nworkers‑th key starting at this worker’s offset
        for key in keys[wid::nworkers]:
            raw = txn.get(key)
            data = pyg2_data_transform(pickle.loads(raw))
            yield data

        # 5) Close when done
        env.close()
