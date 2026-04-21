"""
Compute empirical distance thresholds for the two clash-check methods in evaluator.py,
using only catalyst atoms (tags != 2) from the OC20 training LMDB.

Method 1: minimum pairwise distance in Å  →  informs the 0.50 Å fixed cutoff
Method 2: minimum (dist / sum_covalent_radii) ratio  →  informs the 0.50 fraction

Outputs a summary to stdout and writes results/dist_thresholds.txt.

Usage:
    python compute_dist_thresholds.py --lmdb /path/to/data.lmdb [--limit N] [--workers N]
"""

import sys
import pickle
import argparse
import os
from pathlib import Path
from multiprocessing import Pool

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
from ase.data import covalent_radii
from tqdm import tqdm


def _numpy_shim():
    import numpy, numpy.core
    if not hasattr(numpy, "_core"):
        numpy._core = numpy.core
        import pkgutil
        for _, modname, _ in pkgutil.walk_packages(
            numpy.core.__path__, prefix="numpy.core.", onerror=lambda x: None
        ):
            alias = modname.replace("numpy.core", "numpy._core")
            if modname in sys.modules:
                sys.modules[alias] = sys.modules[modname]


def _to_np(x):
    try:
        return np.array(x.cpu().numpy(), dtype=np.float64)
    except AttributeError:
        return np.array(x, dtype=np.float64)


def _process_raw(raw: bytes) -> tuple[float, float] | tuple[None, None]:
    """Unpickle one LMDB entry and compute (min_dist_m1, min_ratio_m2)."""
    _numpy_shim()
    import numpy as np
    from ase import Atoms
    from ase.data import covalent_radii

    try:
        data = pickle.loads(raw)
        tags    = np.array(data.tags,           dtype=int).flatten()
        pos     = np.array(data.pos,            dtype=np.float64)
        numbers = np.array(data.atomic_numbers, dtype=int).flatten()
        cell    = np.array(data.cell,           dtype=np.float64).reshape(3, 3)
    except Exception:
        return None, None

    cat_mask = tags != 2
    if cat_mask.sum() < 2:
        return None, None

    atoms = Atoms(
        numbers=numbers[cat_mask],
        positions=pos[cat_mask],
        cell=cell,
        pbc=True,
    )

    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)

    min_dist = float(dists.min())

    radii     = covalent_radii[numbers[cat_mask]]
    sum_radii = radii[:, None] + radii[None, :]
    sum_radii[sum_radii == 0.0] = np.inf  # undefined covalent radius → skip pair
    # leave sum_radii diagonal finite: inf/finite = inf, avoiding inf/inf = nan
    ratio     = dists / sum_radii
    min_ratio = float(ratio.min())

    return min_dist, min_ratio


def run(lmdb_path: str, limit: int | None, workers: int, out_file: Path) -> None:
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=workers + 1,
    )

    with env.begin() as txn:
        length_entry = txn.get(b"length")
        n_total = pickle.loads(length_entry) if length_entry is not None else env.stat()["entries"]

    n_scan = n_total if limit is None else min(limit, n_total)
    print(f"Scanning {n_scan} / {n_total} entries  |  workers: {workers}")

    print("Reading raw bytes from LMDB...")
    with env.begin() as txn:
        raws = []
        for idx in tqdm(range(n_scan), desc="Reading", ncols=100):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is not None:
                raws.append(raw)
    env.close()

    print(f"Processing {len(raws)} entries in parallel...")
    chunksize = max(1, len(raws) // (workers * 8))
    with Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap(_process_raw, raws, chunksize=chunksize),
            total=len(raws),
            desc="Processing",
            ncols=100,
        ))

    min_dists_m1 = [m1 for m1, _ in results if m1 is not None]
    min_ratios_m2 = [m2 for _, m2 in results if m2 is not None]
    n_skipped = len(raws) - len(min_dists_m1)

    n_valid = len(min_dists_m1)
    print(f"\nValid structures : {n_valid}  |  Skipped : {n_skipped}")

    arr_m1 = np.array(min_dists_m1)
    arr_m2 = np.array(min_ratios_m2)

    stats = {
        "method1_mean_min_dist_A":   float(arr_m1.mean()),
        "method1_median_min_dist_A": float(np.median(arr_m1)),
        "method1_p5_min_dist_A":     float(np.percentile(arr_m1, 5)),
        "method1_p1_min_dist_A":     float(np.percentile(arr_m1, 1)),
        "method1_min_of_mins_A":     float(arr_m1.min()),
        "method2_mean_min_ratio":    float(arr_m2.mean()),
        "method2_median_min_ratio":  float(np.median(arr_m2)),
        "method2_p5_min_ratio":      float(np.percentile(arr_m2, 5)),
        "method2_p1_min_ratio":      float(np.percentile(arr_m2, 1)),
        "method2_min_of_mins_ratio": float(arr_m2.min()),
    }

    lines = [
        "=" * 70,
        "OC20 Catalyst-Atom Distance Threshold Analysis",
        f"LMDB            : {lmdb_path}",
        f"Structures used : {n_valid}",
        f"Structures limit: {limit if limit is not None else 'all'}",
        f"Workers         : {workers}",
        "=" * 70,
        "",
        "METHOD 1 — raw minimum pairwise distance (Å)",
        "  (replaces the fixed '0.50 Å' cutoff in evaluator.py DIST_METHOD=1)",
        f"  mean of per-structure min dist  : {stats['method1_mean_min_dist_A']:.4f} Å",
        f"  median of per-structure min dist: {stats['method1_median_min_dist_A']:.4f} Å",
        f"  5th-percentile                  : {stats['method1_p5_min_dist_A']:.4f} Å",
        f"  1st-percentile                  : {stats['method1_p1_min_dist_A']:.4f} Å",
        f"  global minimum                  : {stats['method1_min_of_mins_A']:.4f} Å",
        "",
        "METHOD 2 — minimum (dist / sum_covalent_radii) ratio",
        "  (replaces the '0.50' fraction in evaluator.py DIST_METHOD=2)",
        f"  mean of per-structure min ratio  : {stats['method2_mean_min_ratio']:.4f}",
        f"  median of per-structure min ratio: {stats['method2_median_min_ratio']:.4f}",
        f"  5th-percentile                   : {stats['method2_p5_min_ratio']:.4f}",
        f"  1st-percentile                   : {stats['method2_p1_min_ratio']:.4f}",
        f"  global minimum                   : {stats['method2_min_of_mins_ratio']:.4f}",
        "",
        "Suggested replacement values (conservative = 5th-percentile):",
        f"  Method 1 threshold : {stats['method1_p5_min_dist_A']:.4f} Å",
        f"  Method 2 fraction  : {stats['method2_p5_min_ratio']:.4f}",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report + "\n")
    print(f"\nSaved → {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute empirical distance thresholds from OC20 catalyst atoms."
    )
    parser.add_argument(
        "--lmdb",
        default="/datasets/oc20/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data_refined.lmdb",
        help="Path to OC20 training LMDB file",
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
        default=os.cpu_count(),
        help="Number of worker processes (default: cpu_count)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results" / "dist_thresholds.txt"),
        help="Output .txt file path",
    )
    args = parser.parse_args()

    run(
        lmdb_path=args.lmdb,
        limit=args.limit,
        workers=args.workers,
        out_file=Path(args.out),
    )


if __name__ == "__main__":
    main()
