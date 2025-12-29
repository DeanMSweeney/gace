from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ase.constraints import FixAtoms
from tqdm import tqdm

from adsflow.utils.IS2S_builders.lmdb_builder import process_single_tar


def process_tar_wrapper(args: Tuple) -> Tuple[str, int, int]:
    """Wrapper function for multiprocessing to unpack arguments."""
    tar_path, selected_frames_df, sid_to_adsorbate, out_dir, entries_per_shard = args
    return process_single_tar(
        tar_path=tar_path,
        selected_frames_df=selected_frames_df,
        sid_to_adsorbate=sid_to_adsorbate,
        out_dir=out_dir,
        entries_per_shard=entries_per_shard,
    )

def main():
    ap = argparse.ArgumentParser(
        description="Build LMDB dataset from selected frames parquet file"
    )
    ap.add_argument(
        "--parquet_file",
        type=str,
        required=True,
        help="Path to selected_frames.parquet file"
    )
    ap.add_argument(
        "--tar_root",
        type=str,
        required=True,
        help="Directory containing per-adsorbate *.tar files"
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for LMDB files"
    )
    ap.add_argument(
        "--entries_per_shard",
        type=int,
        default=100_000,
        help="Number of entries per LMDB shard"
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers"
    )
    args = ap.parse_args()

    parquet_file = Path(args.parquet_file)
    tar_root = Path(args.tar_root)
    out_dir = Path(args.out_dir)

    # Validate inputs
    if not parquet_file.exists():
        print(f"Error: parquet file not found: {parquet_file}", file=sys.stderr)
        sys.exit(1)

    if not tar_root.exists():
        print(f"Error: tar_root not found: {tar_root}", file=sys.stderr)
        sys.exit(1)

    # Read selected frames
    print(f"Reading selected frames from {parquet_file}...", flush=True)
    selected_df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(selected_df)} trajectory selections", flush=True)
    print(f"Columns in parquet: {selected_df.columns.tolist()}", flush=True)

    # Build sid -> adsorbate mapping from parquet
    sid_to_adsorbate = {}
    for _, row in selected_df.iterrows():
        sid_to_adsorbate[int(row['sid'])] = str(row['adsorbate'])
    print(f"Mapped {len(sid_to_adsorbate)} unique sids to adsorbates", flush=True)

    # Get list of tar files
    tar_paths = sorted(tar_root.glob("*.tar"))
    if not tar_paths:
        print(f"Error: No .tar files found in {tar_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tar_paths)} tar files to process with {args.num_workers} workers", flush=True)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    process_args = [
        (tar_path, selected_df, sid_to_adsorbate, out_dir, args.entries_per_shard)
        for tar_path in tar_paths
    ]

    # Process tar files in parallel with progress bar
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_tar_wrapper, process_args),
            total=len(process_args),
            desc="Processing tar files",
            unit="tar"
        ))

    # Aggregate results
    total_traj = sum(n_traj for _, n_traj, _ in results)
    total_frames = sum(n_frames for _, _, n_frames in results)

    # Print summary
    print("\n=== Summary ===", flush=True)
    print(f"Total trajectories processed: {total_traj}", flush=True)
    print(f"Total frames extracted: {total_frames}", flush=True)
    print(f"LMDB files written to: {out_dir}", flush=True)

    # Count shards
    lmdb_files = sorted(out_dir.glob("*.lmdb"))
    print(f"Number of LMDB shards: {len(lmdb_files)}", flush=True)


if __name__ == "__main__":
    main()
