#!/usr/bin/env python3
"""
Stream OC20 per-adsorbate IS2RE/S trajectory tarballs and build sharded metrics.

Output:
<out_root>/<adsorbate>/metrics/part-00000.parquet, part-00001.parquet, ...

Each row corresponds to one frame.
"""

from __future__ import annotations

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

from adsflow.utils.IS2S_builders.frame_metrics import process_single_tar

SID_RE = re.compile(r"(?:sid[_\-]?|^)(\d+)(?:\D|$)", re.IGNORECASE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True, help="Directory containing per-adsorbate *.tar files")
    ap.add_argument("--rows_per_shard", type=int, default=2_000_000, help="Frames (rows) per parquet shard")
    ap.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = in_root
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted(in_root.glob("*.tar"))
    if not tar_paths:
        print(f"No .tar files found in {in_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tar_paths)} tar files with {args.num_workers} workers", flush=True)

    # Prepare arguments for parallel processing
    process_args = [(tar_path, out_root, args.rows_per_shard) for tar_path in tar_paths]

    # Process tar files in parallel
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_single_tar, process_args)

    # Print summary
    print("\n=== Summary ===", flush=True)
    for adsorbate, n_members, n_traj_ok, n_traj_skip in results:
        print(
            f"  {adsorbate}: members={n_members} ok={n_traj_ok} skipped={n_traj_skip}",
            flush=True,
        )

if __name__ == "__main__":
    main()
