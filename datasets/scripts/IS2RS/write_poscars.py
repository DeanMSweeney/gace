"""
Write POSCAR files for structures whose data.sid matches a target set.
Output: datasets/scripts/is2rs/poscars/{sid}_relaxed.vasp
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

LMDB_PATH = "/datasets/oc20//full/is2res_train_val_test_lmdbs/data/is2re/all/train/data_refined.lmdb"
OUT_DIR = Path(__file__).parent / "poscars"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SIDS = {
    2472718,
    1747243,
    1372820,
    2323648,
    533810,
    1393558,
    2204646,
    1663491,
    1981203,
    533248,
}

env = lmdb.open(
    LMDB_PATH,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

found = set()
with env.begin() as txn:
    # get total number of entries
    length_entry = txn.get(b"length")
    if length_entry is not None:
        n_total = pickle.loads(length_entry)
    else:
        n_total = env.stat()["entries"]

    print(f"Scanning {n_total} entries for {len(TARGET_SIDS)} target sids...")

    for idx in range(n_total):
        if len(found) == len(TARGET_SIDS):
            break

        raw = txn.get(str(idx).encode("ascii"))
        if raw is None:
            continue

        data = pickle.loads(raw)

        # sid may be a plain int, a tensor, or a numpy scalar
        sid = data.sid
        try:
            sid = int(sid)
        except Exception:
            continue

        if sid in TARGET_SIDS and sid not in found:
            out_path = str(OUT_DIR / f"{sid}_relaxed.vasp")
            write_strucs(data, gen_POS=None, rlxd_POS=out_path)
            found.add(sid)
            print(f"  [{len(found)}/{len(TARGET_SIDS)}] sid={sid} (lmdb idx={idx}) -> {out_path}")

env.close()

missing = TARGET_SIDS - found
if missing:
    print(f"\nWarning: {len(missing)} sid(s) not found in dataset: {missing}")
else:
    print(f"\nAll {len(TARGET_SIDS)} structures written to {OUT_DIR}")
