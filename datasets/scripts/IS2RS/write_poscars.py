"""
Write POSCAR files for the first 10 relaxed structures from the IS2RE refined LMDB.
Output: datasets/scripts/is2rs/poscars/{idx}_relaxed.vasp
"""

import sys
import pickle
from pathlib import Path

# Numpy compatibility shim (data may be pickled with numpy>=1.25)
import numpy, numpy.core
if not hasattr(numpy, '_core'):
    numpy._core = numpy.core
    import pkgutil
    for _, modname, _ in pkgutil.walk_packages(numpy.core.__path__, prefix='numpy.core.', onerror=lambda x: None):
        alias = modname.replace('numpy.core', 'numpy._core')
        if modname in sys.modules:
            sys.modules[alias] = sys.modules[modname]

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import lmdb
from adsflow.utils.atoms import write_strucs

LMDB_PATH = "/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data_refined.lmdb"
N = 10
OUT_DIR = Path(__file__).parent / "poscars"
OUT_DIR.mkdir(exist_ok=True)

env = lmdb.open(
    LMDB_PATH,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

with env.begin() as txn:
    for idx in range(N):
        raw = txn.get(str(idx).encode("ascii"))
        if raw is None:
            print(f"No entry at index {idx}, stopping.")
            break

        data = pickle.loads(raw)
        out_path = str(OUT_DIR / f"{idx}_relaxed.vasp")
        write_strucs(data, gen_POS=None, rlxd_POS=out_path)
        print(f"[{idx}] wrote {out_path}")

env.close()
