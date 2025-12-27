#!/usr/bin/env python3
"""
Build sharded LMDB dataset from selected frames parquet file.

Reads the selected_frames.parquet file (created by select_frames.py) and
extracts the corresponding frames from the trajectory tar files to build
an LMDB dataset compatible with lmdb_dataset.py.

The LMDB dataset format matches OC20 structure with the following fields:
- pos: atomic positions (N, 3)
- pos_relaxed: relaxed positions from last frame (N, 3)
- cell: unit cell (3, 3)
- atomic_numbers: atomic numbers (N,)
- natoms: number of atoms per structure (scalar)
- force: forces on atoms (N, 3)
- fixed: boolean mask for fixed atoms (N,)
- tags: atom tags (N,)
- y: energy of current frame (scalar)
- y_init: initial energy from first frame (scalar)
- y_relaxed: relaxed energy from last frame (scalar)
- sid: system id (scalar)
- fid: frame index (scalar)

Note: edge_index and cell_offsets are not included - these can be computed
on-the-fly during training if needed.
"""

from __future__ import annotations

import argparse
import io
import lmdb
import lzma
import pickle
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.constraints import FixAtoms
from tqdm import tqdm

# Regex for extracting sid from filenames
SID_RE = re.compile(r"(?:sid[_\-]?|^)(\d+)(?:\D|$)", re.IGNORECASE)


def infer_adsorbate_name(tar_path: Path) -> str:
    """Infer adsorbate folder name from tar filename."""
    return tar_path.stem


def infer_sid_from_member(member_name: str) -> Optional[int]:
    """Extract sid from tar member filename."""
    base = Path(member_name).name
    m = SID_RE.search(base)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", base)
    return int(m2.group(1)) if m2 else None


def read_extxyz_xz_from_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> List:
    """
    Stream member bytes -> lzma decompress -> ASE read all frames.
    """
    f = tar.extractfile(member)
    if f is None:
        return []
    raw = f.read()
    decompressed = lzma.decompress(raw).decode('utf-8')
    atoms_list = ase_read(io.StringIO(decompressed), format="extxyz", index=":")
    if atoms_list is None:
        return []
    if isinstance(atoms_list, list):
        return atoms_list
    return [atoms_list]


def get_fixed_mask(atoms) -> np.ndarray:
    """Get boolean mask where True = fixed atoms, False = free atoms."""
    mask = np.zeros(len(atoms), dtype=bool)
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            mask[c.index] = True
    return mask


def atoms_to_lmdb_data(
    atoms,
    atoms_relaxed,
    atoms_init,
    sid: int,
    frame_idx: int,
) -> Dict:
    """
    Convert ASE Atoms object to LMDB-compatible dictionary.

    Args:
        atoms: Current frame Atoms object
        atoms_relaxed: Relaxed (last frame) Atoms object
        atoms_init: Initial (first frame) Atoms object
        sid: System ID
        frame_idx: Frame index in trajectory

    Returns:
        Dictionary with OC20-style LMDB fields
    """
    data = {
        # Positions
        "pos": atoms.get_positions().astype(np.float32),
        "pos_init": atoms_init.get_positions().astype(np.float32),

        # Cell
        "cell": atoms.get_cell().array.astype(np.float32),

        # Atomic information
        "atomic_numbers": atoms.get_atomic_numbers().astype(np.int64),
        "natoms": len(atoms),

        # Forces and constraints
        "force": atoms.get_forces().astype(np.float32),
        "fixed": get_fixed_mask(atoms),
        "tags": atoms.get_tags().astype(np.int64),

        # Energies
        "y": float(atoms.get_potential_energy()),
        "y_init": float(atoms_init.get_potential_energy()),
        "y_relaxed": float(atoms_relaxed.get_potential_energy()),

        # Metadata
        "sid": int(sid),
        "fid": int(frame_idx),
    }

    return data


@dataclass
class LmdbWriter:
    """Writer for sharded LMDB files."""
    out_dir: Path
    entries_per_shard: int = 10_000
    map_size: int = 1099511627776  # 1TB

    _data: List[Dict] = None
    _shard_id: int = 0
    _global_index: int = 0

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._data = []
        self._shard_id = self._find_next_shard_id()

    def _find_next_shard_id(self) -> int:
        """Resume-friendly: continue after max existing shard id."""
        existing = sorted(self.out_dir.glob("data_*.lmdb"))
        if not existing:
            return 0
        max_id = -1
        for p in existing:
            m = re.search(r"data_(\d+)\.lmdb$", p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def add(self, data: Dict) -> None:
        """Add a data entry to the current shard."""
        self._data.append(data)
        if len(self._data) >= self.entries_per_shard:
            self.flush()

    def flush(self) -> None:
        """Write accumulated data to a new LMDB shard."""
        if not self._data:
            return

        shard_path = self.out_dir / f"data_{self._shard_id:04d}.lmdb"

        # Create LMDB environment
        env = lmdb.open(
            str(shard_path),
            map_size=self.map_size,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        # Write data
        with env.begin(write=True) as txn:
            for i, entry in enumerate(self._data):
                txn.put(
                    f"{i}".encode("ascii"),
                    pickle.dumps(entry, protocol=-1)
                )
            # Store length for fast dataset initialization
            txn.put(b"length", pickle.dumps(len(self._data), protocol=-1))

        env.close()

        print(f"[write] {shard_path}  entries={len(self._data)}", flush=True)
        self._global_index += len(self._data)
        self._data.clear()
        self._shard_id += 1

    def close(self) -> None:
        """Flush any remaining data and close."""
        self.flush()


def build_sid_to_adsorbate_map(metrics_dir: Path) -> Dict[int, str]:
    """
    Build mapping from sid to adsorbate by reading trajectory metrics.

    Args:
        metrics_dir: Root directory containing adsorbate subdirectories with metrics

    Returns:
        Dictionary mapping sid -> adsorbate name
    """
    sid_to_ads = {}

    # Find all metrics parquet files
    # Structure: metrics_dir/<adsorbate>/metrics/*.parquet
    for ads_dir in metrics_dir.iterdir():
        if not ads_dir.is_dir():
            continue

        metrics_path = ads_dir / "metrics"
        if not metrics_path.exists():
            continue

        adsorbate = ads_dir.name

        # Read all parquet files in this adsorbate's metrics folder
        for pq_file in metrics_path.glob("*.parquet"):
            if pq_file.name.startswith("selected"):
                continue  # Skip selected_frames.parquet

            df = pd.read_parquet(pq_file)
            if 'sid' in df.columns:
                for sid in df['sid'].unique():
                    sid_to_ads[int(sid)] = adsorbate

    return sid_to_ads


def process_single_tar(
    tar_path: Path,
    selected_frames_df: pd.DataFrame,
    sid_to_adsorbate: Dict[int, str],
    out_dir: Path,
    entries_per_shard: int,
) -> Tuple[str, int, int]:
    """
    Process a single tar file and extract selected frames.

    Args:
        tar_path: Path to tar file
        selected_frames_df: DataFrame with selected frames
        sid_to_adsorbate: Mapping from sid to adsorbate name
        out_dir: Output directory for LMDB files
        entries_per_shard: Number of entries per LMDB shard

    Returns:
        (adsorbate, n_trajectories_processed, n_frames_extracted)
    """
    adsorbate = infer_adsorbate_name(tar_path)

    # Create mapping: sid -> list of frame indices for this adsorbate
    sid_to_frames = {}
    for _, row in selected_frames_df.iterrows():
        sid = int(row['sid'])

        # Check if this sid belongs to current adsorbate
        if sid_to_adsorbate.get(sid) != adsorbate:
            continue

        frames = row['selected_frames']
        if isinstance(frames, list):
            sid_to_frames[sid] = frames
        else:
            # Handle case where it might be a single value
            sid_to_frames[sid] = [frames]

    if len(sid_to_frames) == 0:
        print(f"[skip] No selected frames for adsorbate '{adsorbate}'", flush=True)
        return (adsorbate, 0, 0)

    print(f"[processing] adsorbate='{adsorbate}'  sids={len(sid_to_frames)}  frames={sum(len(v) for v in sid_to_frames.values())}", flush=True)

    writer = LmdbWriter(out_dir=out_dir, entries_per_shard=entries_per_shard)

    n_traj_ok = 0
    n_frames_extracted = 0

    with tarfile.open(tar_path, "r") as tar:
        # Iterate through tar members
        for member in tar:
            if not member.isfile() or not member.name.endswith(".extxyz.xz"):
                continue

            sid = infer_sid_from_member(member.name)
            if sid is None or sid not in sid_to_frames:
                continue

            # Read full trajectory
            traj_atoms = read_extxyz_xz_from_tar_member(tar, member)
            if not traj_atoms:
                continue

            n_traj_ok += 1

            # Get the frames we need for this sid
            frames_to_extract = sid_to_frames[sid]

            # Get relaxed structure (last frame)
            atoms_relaxed = traj_atoms[-1]
            # Get initial structure (first frame)
            atoms_init = traj_atoms[0]

            # Extract selected frames
            for frame_idx in frames_to_extract:
                if frame_idx >= len(traj_atoms):
                    print(f"[warning] frame_idx={frame_idx} out of range for sid={sid} (max={len(traj_atoms)-1})", flush=True)
                    continue

                atoms = traj_atoms[frame_idx]

                # Convert to LMDB format
                data = atoms_to_lmdb_data(
                    atoms=atoms,
                    atoms_relaxed=atoms_relaxed,
                    atoms_init=atoms_init,
                    sid=sid,
                    frame_idx=frame_idx,
                )

                writer.add(data)
                n_frames_extracted += 1

    writer.close()

    print(f"[done] adsorbate='{adsorbate}' traj_ok={n_traj_ok} frames_extracted={n_frames_extracted}", flush=True)

    return (adsorbate, n_traj_ok, n_frames_extracted)


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
        "--metrics_root",
        type=str,
        required=False,
        help="Directory containing trajectory metrics (for sid->adsorbate mapping). Defaults to tar_root."
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
        default=10_000,
        help="Number of entries per LMDB shard"
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential processing)"
    )
    args = ap.parse_args()

    parquet_file = Path(args.parquet_file)
    tar_root = Path(args.tar_root)
    metrics_root = Path(args.metrics_root) if args.metrics_root else tar_root
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

    # Check if adsorbate column exists in parquet
    if 'adsorbate' in selected_df.columns:
        print(f"Using adsorbate information from parquet file", flush=True)
        # Build sid -> adsorbate mapping from parquet
        sid_to_adsorbate = {}
        for _, row in selected_df.iterrows():
            sid_to_adsorbate[int(row['sid'])] = str(row['adsorbate'])
        print(f"Mapped {len(sid_to_adsorbate)} unique sids to adsorbates", flush=True)
    else:
        # Fall back to building mapping from trajectory metrics
        print(f"Adsorbate column not found in parquet, building sid->adsorbate mapping from {metrics_root}...", flush=True)
        sid_to_adsorbate = build_sid_to_adsorbate_map(metrics_root)
        print(f"Mapped {len(sid_to_adsorbate)} unique sids to adsorbates", flush=True)

    # Get list of tar files
    tar_paths = sorted(tar_root.glob("*.tar"))
    if not tar_paths:
        print(f"Error: No .tar files found in {tar_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tar_paths)} tar files to process", flush=True)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process tar files
    if args.num_workers > 1:
        print(f"Processing with {args.num_workers} parallel workers...", flush=True)
        # For parallel processing, we need to be careful with LMDB writing
        # Each worker should write to separate shards
        # This is complex, so for now we'll use sequential processing
        print("[warning] Parallel processing not yet implemented, using sequential", flush=True)
        args.num_workers = 1

    # Sequential processing
    total_traj = 0
    total_frames = 0

    for tar_path in tqdm(tar_paths, desc="Processing tar files"):
        _, n_traj, n_frames = process_single_tar(
            tar_path=tar_path,
            selected_frames_df=selected_df,
            sid_to_adsorbate=sid_to_adsorbate,
            out_dir=out_dir,
            entries_per_shard=args.entries_per_shard,
        )
        total_traj += n_traj
        total_frames += n_frames

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
