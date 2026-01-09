#!/usr/bin/env python3
"""
Stream OC20 per-adsorbate IS2RE/S trajectory tarballs and build metrics.

Output:
<out_dir>/{adsorbate}.parquet

Each row corresponds to one frame.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

from adsflow.utils.IS2S_builders.frame_metrics import process_single_tar

SID_RE = re.compile(r"(?:sid[_\-]?|^)(\d+)(?:\D|$)", re.IGNORECASE)

def process_single_tar_safe(args_tuple):
    """Wrapper with error handling for worker processes."""
    try:
        result = process_single_tar(args_tuple)
        return result
    except Exception as e:
        tar_path = args_tuple[0] if args_tuple else "unknown"
        error_msg = f"ERROR processing {tar_path}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        return (str(tar_path), 0, 0, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", type=str, required=True, help="Directory containing per-adsorbate *.tar files")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory containing to store new data")
    ap.add_argument("--rows_per_shard", type=int, default=2_000_000, help="Frames (rows) per parquet shard")
    ap.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted(traj_dir.glob("*.tar"))
    if not tar_paths:
        print(f"No .tar files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tar_paths)} tar files with {args.num_workers} workers", flush=True)

    # Prepare arguments for parallel processing
    process_args = [(tar_path, out_dir, args.rows_per_shard) for tar_path in tar_paths]

    # Process tar files in parallel
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_single_tar_safe, process_args)

    # Print summary
    print("\n=== Summary ===", flush=True)
    for adsorbate, n_members, n_traj_ok, n_traj_skip in results:
        print(
            f"  {adsorbate}: members={n_members} ok={n_traj_ok} skipped={n_traj_skip}",
            flush=True,
        )

if __name__ == "__main__":
    main()
