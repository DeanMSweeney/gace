"""
Collect OC20 samples whose adsorbate is a single oxygen or nitrogen atom,
building an ASE Atoms object for each and writing them as .traj files into an
output directory.

Adsorbate atoms are identified by ``tags == 2`` (slab/surface = other tags).
A sample matches only when the adsorbate is a *single* atom that is oxygen
(Z=8) or nitrogen (Z=7).

Each saved Atoms object carries its source LMDB index and the matched adsorbate
element in ``atoms.info`` so downstream code can trace it back.

Work is split across processes; each worker opens its own LMDB handle and writes
its matched .traj files independently.

Usage:
    python collect_ON_adsorbates.py --lmdb /path/to/data.lmdb --out-dir /path/to/dir \
        [--limit N] [--workers W]
"""

import sys
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Numpy compatibility shim (data pickled with numpy>=1.25)
import numpy, numpy.core  # noqa: E401
if not hasattr(numpy, "_core"):
    numpy._core = numpy.core
    import pkgutil
    for _, modname, _ in pkgutil.walk_packages(
        numpy.core.__path__, prefix="numpy.core.", onerror=lambda x: None
    ):
        alias = modname.replace("numpy.core", "numpy._core")
        if modname in sys.modules:
            sys.modules[alias] = sys.modules[modname]

import lmdb
import numpy as np
from ase import Atoms
from ase.io import write
from tqdm import tqdm


# Atomic numbers of the adsorbate elements we want to keep.
TARGET_Z = {7: "N", 8: "O"}


def _to_np(x):
    """Convert tensor / numpy array / list to plain numpy float64 array."""
    try:
        return np.array(x.cpu().numpy(), dtype=np.float64)
    except AttributeError:
        return np.array(x, dtype=np.float64)


def build_atoms(data, idx: int) -> Atoms | None:
    """
    Return an ``ase.Atoms`` for this structure if its adsorbate is a single
    O or N atom, else ``None``.
    """
    try:
        tags    = _to_np(data.tags).astype(int).flatten()
        pos     = _to_np(data.pos)
        numbers = _to_np(data.atomic_numbers).astype(int).flatten()
        cell    = _to_np(data.cell).reshape(3, 3)
    except AttributeError:
        return None

    ads_mask = tags == 2
    ads_numbers = numbers[ads_mask]

    # Single-atom adsorbate that is O or N.
    if ads_numbers.size != 1 or int(ads_numbers[0]) not in TARGET_Z:
        return None

    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
    atoms.set_tags(tags)
    atoms.info["lmdb_index"] = idx
    atoms.info["ads_element"] = TARGET_Z[int(ads_numbers[0])]
    atoms.info["n_ads_atoms"] = 1
    return atoms


# Globals initialised once per worker process.
_ENV = None
_LMDB_PATH = None
_OUT_DIR = None


def _init_worker(lmdb_path: str, out_dir: str) -> None:
    global _ENV, _LMDB_PATH, _OUT_DIR
    _LMDB_PATH = lmdb_path
    _OUT_DIR = Path(out_dir)
    _ENV = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _process_chunk(index_range: tuple[int, int]) -> dict[str, int]:
    """Process [start, stop) of LMDB indices, writing matched .traj files."""
    start, stop = index_range
    counts: dict[str, int] = {"_missing": 0}
    with _ENV.begin() as txn:
        for idx in range(start, stop):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                counts["_missing"] += 1
                continue

            atoms = build_atoms(pickle.loads(raw), idx)
            if atoms is None:
                continue

            elem = atoms.info["ads_element"]
            write(str(_OUT_DIR / f"{idx:08d}_{elem}.traj"), atoms)
            counts[elem] = counts.get(elem, 0) + 1
    return counts


def run(lmdb_path: str, limit: int | None, workers: int, out_dir: Path) -> None:
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin() as txn:
        length_entry = txn.get(b"length")
        n_total = pickle.loads(length_entry) if length_entry is not None else env.stat()["entries"]
    env.close()

    n_scan = n_total if limit is None else min(limit, n_total)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split the index range into one chunk per worker (a few chunks per worker
    # keeps the tqdm bar responsive and load roughly balanced).
    n_chunks = max(workers * 4, 1)
    bounds = np.linspace(0, n_scan, n_chunks + 1, dtype=int)
    chunks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    print(f"Scanning {n_scan} / {n_total} entries from {lmdb_path}  "
          f"(single-atom O/N adsorbate, {workers} workers)")

    totals: dict[str, int] = {}
    n_missing = 0
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(lmdb_path, str(out_dir)),
    ) as pool:
        for counts in tqdm(
            pool.imap_unordered(_process_chunk, chunks),
            total=len(chunks),
            desc="Chunks",
            ncols=100,
        ):
            n_missing += counts.pop("_missing", 0)
            for elem, c in counts.items():
                totals[elem] = totals.get(elem, 0) + c

    n_matched = sum(totals.values())
    print(f"\nMatched structures : {n_matched}  |  Missing entries : {n_missing}")
    for elem in sorted(totals):
        print(f"  {elem:>4} adsorbate : {totals[elem]}")
    print(f"\nSaved {n_matched} .traj files → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect OC20 samples whose adsorbate is a single oxygen or nitrogen atom."
    )
    parser.add_argument(
        "--lmdb",
        default="/datasets/oc20/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data_refined.lmdb",
        help="Path to OC20 LMDB file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of structures to scan (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: all CPUs)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write per-sample .traj files into",
    )
    args = parser.parse_args()

    run(
        lmdb_path=args.lmdb,
        limit=args.limit,
        workers=max(args.workers, 1),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
