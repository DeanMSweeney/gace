"""
process_lmdb.py
───────────────
Script to preprocess a raw CatFlow LMDB file into the augmented format
required for training.  Supports parallel processing across multiple CPU
cores and an optional limit on the number of samples to process.

Usage
-----
    # Process all samples using all available cores
    python process_lmdb.py --src dataset.lmdb --dst processed.lmdb

    # Process only the first 100 samples using 4 workers
    python process_lmdb.py --src dataset.lmdb --dst processed.lmdb --limit 100 --workers 4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from multiprocessing import Pool

# Numpy compatibility shim (data pickled with numpy>=1.25, redirects _core -> core)
import numpy, numpy.core
if not hasattr(numpy, '_core'):
    numpy._core = numpy.core
    import pkgutil
    for _, modname, _ in pkgutil.walk_packages(numpy.core.__path__, prefix='numpy.core.', onerror=lambda x: None):
        alias = modname.replace('numpy.core', 'numpy._core')
        if modname in sys.modules:
            sys.modules[alias] = sys.modules[modname]

import lmdb
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from gace.utils.catflow_builders.data_processor import process_sample


def _worker(raw: bytes) -> bytes:
    """Top-level wrapper so Pool can pickle the target function."""
    import sys, numpy, numpy.core
    if not hasattr(numpy, '_core'):
        numpy._core = numpy.core
        import pkgutil
        for _, modname, _ in pkgutil.walk_packages(numpy.core.__path__, prefix='numpy.core.', onerror=lambda x: None):
            alias = modname.replace('numpy.core', 'numpy._core')
            if modname in sys.modules:
                sys.modules[alias] = sys.modules[modname]
    return process_sample(raw)


def run(src: Path, dst: Path, limit: int | None, workers: int) -> None:
    # ── Open source LMDB ──────────────────────────────────────────────────────
    src_env = lmdb.open(
        str(src), subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False, max_readers=1,
    )

    if dst is None:
        dst = src.parent / f"{src.stem}_processed.lmdb"

    with src_env.begin() as txn:
        length_entry = txn.get(b"length")
        total = pickle.loads(length_entry) if length_entry is not None else src_env.stat()["entries"]

    num_samples = total if limit is None else min(limit, total)

    print(f"Source  : {src}  ({total} total samples)")
    print(f"Output  : {dst}")
    print(f"Process : {num_samples} samples  |  workers: {workers}")

    # ── Read raw bytes for the requested range ────────────────────────────────
    with src_env.begin() as txn:
        raw_items = []
        for i in range(num_samples):
            key = f"{i}".encode("ascii")
            raw = txn.get(key)
            if raw is not None:
                raw_items.append((i, raw))

    src_env.close()

    # ── Process in parallel ───────────────────────────────────────────────────
    keys   = [item[0] for item in raw_items]
    raws   = [item[1] for item in raw_items]

    with Pool(processes=workers) as pool:
        processed = list(tqdm(
            pool.imap(_worker, raws, chunksize=max(1, len(raws) // (workers * 4))),
            total=len(raws),
            desc="Preprocessing",
        ))

    # ── Write output LMDB (single writer in main process) ────────────────────
    src_env_info = lmdb.open(str(src), subdir=False, readonly=True, lock=False)
    map_size = src_env_info.info()["map_size"] * 4
    src_env_info.close()

    dst_env = lmdb.open(str(dst), subdir=False, readonly=False, map_size=map_size, meminit=False)

    with dst_env.begin(write=True) as txn:
        for idx, data in zip(keys, processed):
            txn.put(f"{idx}".encode("ascii"), data)
        txn.put(b"length", pickle.dumps(num_samples, protocol=4))

    dst_env.close()
    print(f"Done. Wrote {num_samples} samples → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a CatFlow LMDB with lattice params and fractional coords."
    )
    parser.add_argument("--src",     required=True,                   help="Input LMDB file path")
    parser.add_argument("--dst",     default=None,                    help="Output LMDB file path")
    parser.add_argument("--limit",   type=int, default=None,          help="Max samples to process (default: all)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    run(
        src=Path(args.src),
        dst=None if args.dst is None else Path(args.dst),
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
