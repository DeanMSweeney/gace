"""
Assess structural validity of .traj files using CatalystForge._filter.

Iterates every .traj file found recursively under a given root directory and
runs the same filter checks used during training/evaluation to assess whether
each structure is physically valid.  Metrics are collected per-folder and in
aggregate, then printed and written to a .txt file.

Usage:
    python assess_validity.py /path/to/dataset
    python assess_validity.py /path/to/dataset --per-folder
    python assess_validity.py /path/to/dataset --out results.txt
    python assess_validity.py /path/to/dataset --cutoff 0.5 --integrity-threshold 2.0

Each .traj file may contain multiple frames (e.g. a relaxation trajectory).
The *last* frame is evaluated as the candidate structure.  For multi-frame
files, the *first* frame supplies the reference adsorbate positions used by the
ads_integrity_ok check; for single-frame files the atoms' own adsorbate
positions are used (ads_integrity_ok is trivially True in that case).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — add gace/src so that `gace.*` imports resolve regardless of
# whether the package is installed.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gace.utils.catalyst_forge import CatalystForge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cf(catalyst_dist_cutoff: float, ads_integrity_threshold: float) -> CatalystForge:
    """Return a CatalystForge instance with only the attributes needed by _filter."""
    cf = object.__new__(CatalystForge)
    cf.catalyst_dist_cutoff = catalyst_dist_cutoff
    cf.ads_integrity_threshold = ads_integrity_threshold
    return cf


def _ref_ads_pos(frames: list[Atoms]) -> np.ndarray:
    """Return adsorbate positions from the first frame for use as the integrity reference."""
    ref = frames[0]
    ads_mask = ref.get_tags() == 2
    if ads_mask.any():
        return ref.positions[ads_mask]
    return np.zeros((0, 3), dtype=np.float32)


def assess_traj(path: Path, cf: CatalystForge) -> dict | None:
    """
    Assess the final frame of a .traj file.

    Returns a validity dict on success, or None if the file cannot be read /
    is otherwise malformed.
    """
    try:
        frames = ase_read(str(path), index=":")
        if not frames:
            return None
        atoms = frames[-1]
        ref_pos = _ref_ads_pos(frames)
        return cf._filter(atoms, ref_pos)
    except Exception as exc:
        print(f"  WARNING: could not process {path}: {exc}", file=sys.stderr)
        return None


def _aggregate(results: list[dict]) -> dict[str, float]:
    """Compute the pass-rate (mean) for each validity key across all results."""
    keys = [
        "vol_ok", "cat_dist_ok", "cat_atom_types_ok",
        "cat_ads_dist_ok", "ads_dist_ok", "ads_integrity_ok",
        "is_valid", "is_valid_naive",
    ]
    if not results:
        return {k: float("nan") for k in keys}
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


# ---------------------------------------------------------------------------
# Printing / writing
# ---------------------------------------------------------------------------

_KEY_LABELS = {
    "vol_ok":            "Volume ok             (>=0.1 A^3)",
    "cat_dist_ok":       "Cat. dist ok          (cov-radii cutoff)",
    "cat_atom_types_ok": "Cat. atom types ok    (OC20 elements Z22-83)",
    "cat_ads_dist_ok":   "Cat-ads dist ok       (>=0.5 A)",
    "ads_dist_ok":       "Ads. dist ok          (>=0.5 A)",
    "ads_integrity_ok":  "Ads. integrity ok     (vs first frame)",
    "is_valid":          "VALID                 (all strict checks)",
    "is_valid_naive":    "VALID (naive)         (simple dist checks)",
}


def _format_block(header: str, stats: dict[str, float], n_files: int, n_ok: int) -> str:
    W = 70
    lines = [
        "=" * W,
        header,
        "=" * W,
        f"  .traj files found   : {n_files}",
        f"  Successfully parsed : {n_ok}",
        f"  {'Check':<42} {'Pass %':>7}  (Pass / Total)",
        f"  {'-' * 56}",
    ]
    for key, label in _KEY_LABELS.items():
        rate = stats.get(key, float("nan"))
        n_pass = round(rate * n_ok) if not np.isnan(rate) else "?"
        pct_str = f"{rate * 100:6.2f}%" if not np.isnan(rate) else "   n/a"
        lines.append(f"  {label:<42} {pct_str}  ({n_pass} / {n_ok})")
    return "\n".join(lines)


def _format_failure_breakdown(all_results: list[dict]) -> str:
    n_ok = len(all_results)
    if n_ok == 0:
        return ""
    n_valid = sum(1 for r in all_results if r["is_valid"])
    n_invalid = n_ok - n_valid
    checks = ["vol_ok", "cat_dist_ok", "cat_atom_types_ok", "cat_ads_dist_ok", "ads_dist_ok"]
    lines = [
        "=" * 70,
        "Failure mode breakdown  (structures that fail VALID, by root cause)",
        "=" * 70,
    ]
    for key in checks:
        label = _KEY_LABELS[key]
        n_fail = sum(1 for r in all_results if not r[key])
        pct = 100.0 * n_fail / n_ok
        lines.append(f"  {label:<42} {n_fail:>6} failing  ({pct:.1f}%)")
    lines += [
        "",
        f"  Total valid  : {n_valid} / {n_ok}  ({100.0 * n_valid / n_ok:.1f}%)",
        f"  Total invalid: {n_invalid} / {n_ok}  ({100.0 * n_invalid / n_ok:.1f}%)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess structural validity of .traj files using CatalystForge._filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Root directory to search for .traj files (searched recursively).")
    parser.add_argument(
        "--cutoff", type=float, default=0.5,
        help="Catalyst distance cutoff as fraction of sum of covalent radii.",
    )
    parser.add_argument(
        "--integrity-threshold", type=float, default=2.0,
        help="Max allowed deviation in adsorbate pairwise distances vs reference frame (A).",
    )
    parser.add_argument(
        "--per-folder", action="store_true",
        help="Include a validity breakdown for each sub-folder in the output.",
    )
    parser.add_argument(
        "--out", type=str, default="validity_results.txt",
        help="Output .txt file path (relative to this script's directory).",
    )
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(__file__).parent / args.out

    cf = _make_cf(args.cutoff, args.integrity_threshold)

    traj_files = sorted(root.rglob("*.traj"))
    if not traj_files:
        print(f"No .traj files found under {root}")
        sys.exit(0)

    header_line = "\n".join([
        "=" * 70,
        "CatalystForge Validity Assessment",
        f"Dataset path : {root}",
        f"Total .traj  : {len(traj_files)}",
        f"Settings     : cutoff={args.cutoff}, integrity_threshold={args.integrity_threshold} A",
        "=" * 70,
    ])
    print(header_line)

    per_folder: dict[Path, list[dict]] = defaultdict(list)
    all_results: list[dict] = []
    n_errors = 0

    for traj_path in tqdm(traj_files, desc="Assessing"):
        result = assess_traj(traj_path, cf)
        if result is not None:
            per_folder[traj_path.parent].append(result)
            all_results.append(result)
        else:
            n_errors += 1

    n_ok = len(all_results)

    sections: list[str] = [header_line]

    # Per-folder breakdown (optional)
    if args.per_folder:
        for folder in sorted(per_folder):
            results = per_folder[folder]
            stats = _aggregate(results)
            rel = folder.relative_to(root) if folder != root else Path(".")
            block = _format_block(
                header=f"Folder: {rel}",
                stats=stats,
                n_files=len(results),
                n_ok=len(results),
            )
            print("\n" + block)
            sections.append(block)

    # Overall summary
    overall = _aggregate(all_results)
    overall_block = _format_block(
        header="OVERALL SUMMARY",
        stats=overall,
        n_files=len(traj_files),
        n_ok=n_ok,
    )
    print("\n" + overall_block)
    sections.append(overall_block)

    if n_errors:
        err_line = f"\n  Files with errors / unreadable : {n_errors}"
        print(err_line)
        sections.append(err_line)

    # Failure breakdown
    if n_ok > 0:
        breakdown = _format_failure_breakdown(all_results)
        print("\n" + breakdown)
        sections.append(breakdown)

    # Write everything to file
    with open(out_path, "w") as f:
        f.write("\n\n".join(sections) + "\n")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
