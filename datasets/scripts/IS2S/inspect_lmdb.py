#!/usr/bin/env python3
"""
Inspect an LMDB dataset created by build_lmdb_from_parquet.py

Usage:
    python inspect_lmdb.py --lmdb_dir /path/to/lmdb/output
    python inspect_lmdb.py --lmdb_dir /path/to/lmdb/output --num_samples 5
"""

import argparse
import lmdb
import pickle
from pathlib import Path

import numpy as np


def inspect_lmdb(lmdb_dir: Path, num_samples: int = 3):
    """
    Inspect LMDB dataset and print sample entries.

    Args:
        lmdb_dir: Directory containing LMDB files
        num_samples: Number of sample entries to print
    """
    lmdb_files = sorted(lmdb_dir.glob("*.lmdb"))

    if not lmdb_files:
        print(f"No LMDB files found in {lmdb_dir}")
        return

    print(f"Found {len(lmdb_files)} LMDB files:")
    for lmdb_file in lmdb_files:
        print(f"  {lmdb_file.name}")

    print()

    total_entries = 0

    # Inspect each LMDB file
    for i, lmdb_file in enumerate(lmdb_files):
        print(f"\n{'='*60}")
        print(f"LMDB File {i+1}/{len(lmdb_files)}: {lmdb_file.name}")
        print(f"{'='*60}")

        env = lmdb.open(
            str(lmdb_file),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1
        )

        # Get number of entries
        with env.begin() as txn:
            length_entry = txn.get(b"length")
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                num_entries = env.stat()["entries"]

        print(f"Number of entries: {num_entries}")
        total_entries += num_entries

        # Print sample entries
        print(f"\nSample entries (first {min(num_samples, num_entries)}):")

        with env.begin() as txn:
            for j in range(min(num_samples, num_entries)):
                key = f"{j}".encode("ascii")
                raw = txn.get(key)

                if raw is None:
                    print(f"  Entry {j}: Not found")
                    continue

                data = pickle.loads(raw)

                print(f"\n  Entry {j}:")
                print(f"    sid: {data.get('sid')}")
                print(f"    fid (frame_idx): {data.get('fid')}")
                print(f"    natoms: {data.get('natoms')}")
                print(f"    y (energy): {data.get('y'):.6f}")
                print(f"    y_init: {data.get('y_init'):.6f}")
                print(f"    y_relaxed: {data.get('y_relaxed'):.6f}")

                if 'pos' in data:
                    pos = data['pos']
                    print(f"    pos shape: {pos.shape}")

                if 'force' in data:
                    force = data['force']
                    print(f"    force shape: {force.shape}")
                    print(f"    max force: {np.linalg.norm(force, axis=1).max():.6f}")

                if 'atomic_numbers' in data:
                    an = data['atomic_numbers']
                    print(f"    atomic_numbers: {np.unique(an).tolist()}")

                if 'fixed' in data:
                    fixed = data['fixed']
                    print(f"    fixed atoms: {fixed.sum()} / {len(fixed)}")

                print(f"    Keys: {list(data.keys())}")

        env.close()

    print(f"\n{'='*60}")
    print(f"Total entries across all LMDB files: {total_entries}")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Inspect LMDB dataset"
    )
    ap.add_argument(
        "--lmdb_dir",
        type=str,
        required=True,
        help="Directory containing LMDB files"
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of sample entries to print per LMDB file"
    )
    args = ap.parse_args()

    lmdb_dir = Path(args.lmdb_dir)

    if not lmdb_dir.exists():
        print(f"Error: Directory not found: {lmdb_dir}")
        return

    inspect_lmdb(lmdb_dir, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
