#!/usr/bin/env python3
"""
Script to compute Fmax from forces in data_refined.lmdb and store in data_is2s.lmdb.

This script:
1. Reads PyG data objects from data_refined.lmdb
2. Extracts forces from data.force (or data_dict["force"])
3. Computes Fmax = max(||F_i||_2) for each structure
4. Stores the data with Fmax added in data_is2s.lmdb

Usage:
    python compute_fmax.py --input /path/to/data_refined.lmdb --output /path/to/output_dir
"""

from __future__ import annotations

import argparse
import lmdb
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data


def compute_fmax(forces):
    """
    Compute Fmax as the maximum force magnitude across all atoms.

    Args:
        forces: Force array of shape (N, 3) where N is number of atoms

    Returns:
        float: Maximum force magnitude
    """
    # Convert to numpy if torch tensor
    if isinstance(forces, torch.Tensor):
        forces = forces.cpu().numpy()

    # Compute L2 norm for each atom and return max
    force_norms = np.linalg.norm(forces, axis=1)
    Fmax = float(force_norms.max())

    return Fmax


def process_lmdb(input_path: Path, output_dir: Path, entries_per_shard: int = 10_000):
    """
    Read data_refined.lmdb, compute Fmax, and write to data_is2s.lmdb.

    Args:
        input_path: Path to input LMDB file or directory
        output_dir: Output directory for new LMDB files
        entries_per_shard: Number of entries per output LMDB shard
    """
    # Determine input LMDB files
    if input_path.is_file():
        input_lmdbs = [input_path]
    else:
        input_lmdbs = sorted(input_path.glob("*.lmdb"))
        if not input_lmdbs:
            raise ValueError(f"No LMDB files found in {input_path}")

    print(f"Found {len(input_lmdbs)} LMDB file(s) to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize output LMDB writer
    total_processed = 0
    shard_id = 0
    current_shard_data = []

    # Process each input LMDB
    for lmdb_path in input_lmdbs:
        print(f"\nProcessing: {lmdb_path}")

        # Open input LMDB
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Get number of entries
        with env.begin() as txn:
            length_entry = txn.get(b"length")
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                num_entries = env.stat()["entries"]

        print(f"  Entries: {num_entries}")

        # Process each entry
        with env.begin() as txn:
            for i in tqdm(range(num_entries), desc="  Processing entries"):
                key = f"{i}".encode("ascii")
                raw = txn.get(key)

                if raw is None:
                    continue

                # Deserialize
                data = pickle.loads(raw)

                # Extract forces and compute Fmax
                if isinstance(data, dict):
                    # Dictionary format from lmdb_builder
                    if "force" in data:
                        forces = data["force"]
                    else:
                        print(f"  Warning: No 'force' key in entry {i}, skipping")
                        continue

                    # Compute Fmax
                    fmax = compute_fmax(forces)

                    # Add Fmax to data
                    data["Fmax"] = fmax

                elif isinstance(data, Data):
                    # PyG Data object
                    if hasattr(data, "force"):
                        forces = data.force
                    else:
                        print(f"  Warning: No 'force' attribute in entry {i}, skipping")
                        continue

                    # Compute Fmax
                    fmax = compute_fmax(forces)

                    # Add Fmax to data (as a scalar tensor for PyG compatibility)
                    data.Fmax = torch.tensor(fmax)

                else:
                    print(f"  Warning: Unknown data type {type(data)} for entry {i}, skipping")
                    continue

                # Add to current shard
                current_shard_data.append(data)
                total_processed += 1

                # Write shard if full
                if len(current_shard_data) >= entries_per_shard:
                    write_shard(current_shard_data, output_dir, shard_id)
                    current_shard_data = []
                    shard_id += 1

        env.close()

    # Write remaining data
    if current_shard_data:
        write_shard(current_shard_data, output_dir, shard_id)

    print(f"\nProcessing complete!")
    print(f"  Total entries processed: {total_processed}")
    print(f"  Output shards created: {shard_id + 1}")
    print(f"  Output directory: {output_dir}")


def write_shard(data_list, output_dir: Path, shard_id: int, map_size: int = 1099511627776):
    """
    Write a shard of data to LMDB.

    Args:
        data_list: List of data entries (dicts or PyG Data objects)
        output_dir: Output directory
        shard_id: Shard number
        map_size: LMDB map size (default 1TB)
    """
    shard_path = output_dir / f"data_is2s_{shard_id:04d}.lmdb"

    # Create LMDB environment
    env = lmdb.open(
        str(shard_path),
        map_size=map_size,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Write data
    with env.begin(write=True) as txn:
        for i, entry in enumerate(data_list):
            txn.put(
                f"{i}".encode("ascii"),
                pickle.dumps(entry, protocol=-1)
            )
        # Store length for fast dataset initialization
        txn.put(b"length", pickle.dumps(len(data_list), protocol=-1))

    env.close()

    print(f"  [write] {shard_path}  entries={len(data_list)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Fmax from forces and create data_is2s.lmdb"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to data_refined.lmdb file or directory containing LMDB files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where data_is2s.lmdb will be stored"
    )
    parser.add_argument(
        "--entries-per-shard",
        type=int,
        default=1_000_000,
        help="Number of entries per LMDB shard (default: 10000)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    # Validate input path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Input path: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Entries per shard: {args.entries_per_shard}")
    print()

    # Process LMDB
    process_lmdb(input_path, output_dir, args.entries_per_shard)


if __name__ == "__main__":
    main()
