"""
Compute empirical distance thresholds for the two clash-check methods in evaluator.py,
using only catalyst atoms (tags != 2) from the OC20 training LMDB.

Method 1: minimum pairwise distance in Å  →  informs the 0.50 Å fixed cutoff
Method 2: minimum (dist / sum_covalent_radii) ratio  →  informs the 0.50 fraction

Outputs a summary to stdout and writes results/dist_thresholds.txt.

Usage:
    python compute_dist_thresholds.py --lmdb /path/to/data.lmdb [--limit N]
"""

import sys
import pickle
import argparse
from pathlib import Path

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


def _to_np(x):
    """Convert tensor / numpy array / list to plain numpy float64 array."""
    try:
        return np.array(x.cpu().numpy(), dtype=np.float64)
    except AttributeError:
        return np.array(x, dtype=np.float64)


def process_structure(data) -> tuple[float | None, float | None]:
    """
    Return (min_dist_method1, min_ratio_method2) for catalyst atoms only.
    Returns (None, None) if the structure has fewer than 2 catalyst atoms.
    """
    try:
        tags    = _to_np(data.tags).astype(int).flatten()
        pos     = _to_np(data.pos)
        numbers = _to_np(data.atomic_numbers).astype(int).flatten()
        cell    = _to_np(data.cell).reshape(3, 3)
    except AttributeError:
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


def run(lmdb_path: str, limit: int | None, out_file: Path) -> None:
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

    n_scan = n_total if limit is None else min(limit, n_total)
    print(f"Scanning {n_scan} / {n_total} entries from {lmdb_path}")

    min_dists_m1: list[float] = []
    min_ratios_m2: list[float] = []
    n_skipped = 0

    with env.begin() as txn:
        for idx in tqdm(range(n_scan), desc="Processing", ncols=100):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                n_skipped += 1
                continue

            data = pickle.loads(raw)
            m1, m2 = process_structure(data)

            if m1 is None:
                n_skipped += 1
                continue

            min_dists_m1.append(m1)
            min_ratios_m2.append(m2)

    env.close()

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
        "--out",
        default=str(Path(__file__).parent / "dist_thresholds.txt"),
        help="Output .txt file path",
    )
    args = parser.parse_args()

    run(
        lmdb_path=args.lmdb,
        limit=args.limit,
        out_file=Path(args.out),
    )


if __name__ == "__main__":
    main()
