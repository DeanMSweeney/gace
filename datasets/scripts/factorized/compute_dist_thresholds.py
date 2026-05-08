"""
Compute empirical distance thresholds for the two clash-check methods in evaluator.py,
using OC20 training LMDB data.

Method 1: minimum pairwise distance in Å  →  informs the 0.50 Å fixed cutoff
Method 2: minimum (dist / sum_covalent_radii) ratio  →  informs the 0.50 fraction

Three atom-pair groups are analysed:
  • catalyst–catalyst  (tags != 2 vs tags != 2)
  • surface–adsorbate  (tags != 2 vs tags == 2)
  • adsorbate–adsorbate (tags == 2 vs tags == 2)

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


def process_structure(data) -> tuple[
    float | None, float | None,   # cat–cat   min_dist, min_ratio
    float | None, float | None,   # surf–ads  min_dist, min_ratio
    float | None, float | None,   # ads–ads   min_dist, min_ratio
]:
    """
    Return per-structure minimum distances and covalent-ratio distances for three
    atom-pair groups.  Any component is None when the relevant subset is too small.
    """
    try:
        tags    = _to_np(data.tags).astype(int).flatten()
        pos     = _to_np(data.pos)
        numbers = _to_np(data.atomic_numbers).astype(int).flatten()
        cell    = _to_np(data.cell).reshape(3, 3)
    except AttributeError:
        return None, None, None, None, None, None

    cat_mask = tags != 2
    ads_mask = tags == 2
    n_cat = int(cat_mask.sum())
    n_ads = int(ads_mask.sum())

    if n_cat < 2 and n_ads < 2:
        return None, None, None, None, None, None

    # Build full Atoms object so MIC distances cover all pairs in one pass.
    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
    all_dists = atoms.get_all_distances(mic=True)

    all_radii  = covalent_radii[numbers]
    sum_radii  = all_radii[:, None] + all_radii[None, :]
    sum_radii[sum_radii == 0.0] = np.inf

    cat_idx = np.where(cat_mask)[0]
    ads_idx = np.where(ads_mask)[0]

    # ── catalyst–catalyst ──────────────────────────────────────────────────
    cat_min_dist = cat_min_ratio = None
    if n_cat >= 2:
        d = all_dists[np.ix_(cat_idx, cat_idx)].copy()
        np.fill_diagonal(d, np.inf)
        cat_min_dist  = float(d.min())
        cat_min_ratio = float((d / sum_radii[np.ix_(cat_idx, cat_idx)]).min())

    # ── surface–adsorbate (cross distances, no diagonal to mask) ──────────
    surf_ads_min_dist = surf_ads_min_ratio = None
    if n_cat >= 1 and n_ads >= 1:
        d = all_dists[np.ix_(cat_idx, ads_idx)]
        surf_ads_min_dist  = float(d.min())
        surf_ads_min_ratio = float((d / sum_radii[np.ix_(cat_idx, ads_idx)]).min())

    # ── adsorbate–adsorbate ───────────────────────────────────────────────
    ads_min_dist = ads_min_ratio = None
    if n_ads >= 2:
        d = all_dists[np.ix_(ads_idx, ads_idx)].copy()
        np.fill_diagonal(d, np.inf)
        ads_min_dist  = float(d.min())
        ads_min_ratio = float((d / sum_radii[np.ix_(ads_idx, ads_idx)]).min())

    return cat_min_dist, cat_min_ratio, surf_ads_min_dist, surf_ads_min_ratio, ads_min_dist, ads_min_ratio


def _stats_block(label: str, arr_dist: np.ndarray, arr_ratio: np.ndarray) -> list[str]:
    def _fmt_dist(arr: np.ndarray, metric: str) -> list[str]:
        return [
            f"  mean of per-structure min {metric}  : {arr.mean():.4f}",
            f"  median of per-structure min {metric}: {np.median(arr):.4f}",
            f"  5th-percentile                   : {np.percentile(arr, 5):.4f}",
            f"  1st-percentile                   : {np.percentile(arr, 1):.4f}",
            f"  global minimum                   : {arr.min():.4f}",
        ]

    lines = [
        "",
        f"{label}",
        f"  Structures with valid pairs: {len(arr_dist)}",
        "",
        f"  METHOD 1 — raw minimum pairwise distance (Å)",
        *_fmt_dist(arr_dist, "dist (Å)"),
        "",
        f"  METHOD 2 — minimum (dist / sum_covalent_radii) ratio",
        *_fmt_dist(arr_ratio, "ratio   "),
    ]
    return lines


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

    cat_dists:      list[float] = []
    cat_ratios:     list[float] = []
    surf_ads_dists: list[float] = []
    surf_ads_ratios: list[float] = []
    ads_dists:      list[float] = []
    ads_ratios:     list[float] = []
    n_skipped = 0

    with env.begin() as txn:
        for idx in tqdm(range(n_scan), desc="Processing", ncols=100):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                n_skipped += 1
                continue

            data = pickle.loads(raw)
            cd, cr, sad, sar, ad, ar = process_structure(data)

            if cd is None and sad is None and ad is None:
                n_skipped += 1
                continue

            if cd  is not None: cat_dists.append(cd);       cat_ratios.append(cr)
            if sad is not None: surf_ads_dists.append(sad); surf_ads_ratios.append(sar)
            if ad  is not None: ads_dists.append(ad);       ads_ratios.append(ar)

    env.close()

    print(f"\nValid structures : {len(cat_dists)}  |  Skipped : {n_skipped}")

    arr_cat_d  = np.array(cat_dists)
    arr_cat_r  = np.array(cat_ratios)
    arr_sad_d  = np.array(surf_ads_dists)
    arr_sad_r  = np.array(surf_ads_ratios)
    arr_ads_d  = np.array(ads_dists)
    arr_ads_r  = np.array(ads_ratios)

    lines = [
        "=" * 70,
        "OC20 Distance Threshold Analysis",
        f"LMDB            : {lmdb_path}",
        f"Structures limit: {limit if limit is not None else 'all'}",
        "=" * 70,
        *_stats_block("CATALYST–CATALYST  (tags != 2 vs tags != 2)", arr_cat_d, arr_cat_r),
        *_stats_block("SURFACE–ADSORBATE  (tags != 2 vs tags == 2)", arr_sad_d, arr_sad_r),
        *_stats_block("ADSORBATE–ADSORBATE (tags == 2 vs tags == 2)", arr_ads_d, arr_ads_r),
        "",
        "Suggested thresholds (conservative = 5th-percentile of catalyst–catalyst):",
        f"  Method 1 threshold : {np.percentile(arr_cat_d, 5):.4f} Å",
        f"  Method 2 fraction  : {np.percentile(arr_cat_r, 5):.4f}",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report + "\n")
    print(f"\nSaved → {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute empirical distance thresholds from OC20 atoms."
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
