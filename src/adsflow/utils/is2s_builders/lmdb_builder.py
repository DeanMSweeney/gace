"""
Build sharded LMDB dataset from selected frames parquet file.

This script reads the selected_frames.parquet file (created by select_frames.py) and
extracts the corresponding frames from the trajectory tar files to build an LMDB dataset.

Key features:
- Iterates through .tar files by adsorbate number
- For each adsorbate, processes all sids with that ads_num from the parquet
- Each entry includes: positions, cell, fixed atoms, Fmax, dEmax, dSmax, sid, fid, pos_init
- All frames include potential energies for consistent batching

LMDB dataset fields:
- pos: atomic positions (N, 3)
- pos_init: initial positions from frame 0 (N, 3)
- cell: unit cell (3, 3)
- atomic_numbers: atomic numbers (N,)
- natoms: number of atoms per structure (scalar)
- fixed: boolean mask for fixed atoms (N,)
- tags: atom tags (N,)
- Fmax: maximum force from parquet (scalar)
- dEmax: energy difference from parquet (scalar)
- dSmax: max position difference from parquet (scalar)
- sid: system id (scalar)
- fid: frame index (scalar)
- y_init: energy of initial frame (scalar)
- y_relaxed: relaxed energy from last frame (scalar)
"""

from __future__ import annotations

import io
import lmdb
import lzma
import pickle
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.constraints import FixAtoms

from adsflow.utils.invariance import pbc_aware_delta

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
            mask[c.index] = True # 1 = frozen, 0 = free 
    return mask

def atoms_to_lmdb_data(
    atoms,
    atoms_init,
    sid: int,
    frame_idx: int,
    Fmax: float,
    dE: float,
    dSmax: float,
) -> Dict:
    """
    Convert ASE Atoms object to LMDB-compatible dictionary.

    Args:
        atoms: Current frame Atoms object
        atoms_init: Initial (first frame) Atoms object
        atoms_relaxed: Relaxed (last frame) Atoms object
        sid: System ID
        frame_idx: Frame index in trajectory
        Fmax: Maximum force from parquet
        dEmax: Energy difference from parquet (dE)
        dSmax: Max position difference from parquet

    Returns:
        Dictionary with LMDB fields including y_init and y_relaxed
    """
    pos = atoms_init.get_positions().astype(np.float32)
    pos_relaxed = atoms.get_positions().astype(np.float32)
    cell = atoms.get_cell().array.astype(np.float32)

    velocity = pbc_aware_delta(
        delta=pos_relaxed - pos,
        cell=cell,
    )  # outputs a torch.Tensor

    data = {
        # Positions
        "pos": pos,
        "pos_relaxed": pos_relaxed,

        # Velocity
        "velocity": velocity.cpu().numpy().astype(np.float32),

        # Cell
        "cell": cell,

        # Atomic information
        "atomic_numbers": atoms.get_atomic_numbers().astype(np.int64),
        "natoms": len(atoms),

        # Constraints
        "fixed": get_fixed_mask(atoms),
        "tags": atoms.get_tags().astype(np.int64),

        # Metrics from parquet
        "Fmax": float(Fmax),
        "dE": float(dE),
        "dSmax": float(dSmax),

        # Metadata
        "sid": int(sid),
        "fid": int(frame_idx),

        # Forces and energies for all frames
        "y_init": float(atoms_init.get_potential_energy()),
        "y_relaxed": float(atoms.get_potential_energy()),
    }

    return data

@dataclass
class LmdbWriter:
    """Writer for sharded LMDB files."""
    out_dir: Path
    entries_per_shard: int = 10_000
    map_size: int = 1099511627776  # 1TB
    prefix: str = "data"  # Prefix for shard filenames

    _data: List[Dict] = field(default_factory=list)
    _shard_id: int = 0
    _global_index: int = 0

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._shard_id = self._find_next_shard_id()

    def _find_next_shard_id(self) -> int:
        """Resume-friendly: continue after max existing shard id."""
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.lmdb"))
        if not existing:
            return 0
        max_id = -1
        for p in existing:
            m = re.search(rf"{re.escape(self.prefix)}_(\d+)\.lmdb$", p.name)
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

        shard_path = self.out_dir / f"{self.prefix}_{self._shard_id:04d}.lmdb"

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

    # Create mapping: sid -> (list of frame indices, Fmax values, dE values, dSmax values)
    sid_to_data = {}
    for _, row in selected_frames_df.iterrows():
        sid = int(row['sid'])

        # Check if this sid belongs to current adsorbate in parquet file 
        if sid_to_adsorbate.get(sid) != adsorbate:
            continue

        frames = row['selected_frames']
        fmax_vals = row['Fmax']
        de_vals = row['dE']
        dsmax_vals = row['dSmax']

        # Ensure they are lists (convert numpy arrays to lists)
        if hasattr(frames, 'tolist'):
            frames = frames.tolist()
        elif not isinstance(frames, list):
            frames = [frames]

        if hasattr(fmax_vals, 'tolist'):
            fmax_vals = fmax_vals.tolist()
        elif not isinstance(fmax_vals, list):
            fmax_vals = [fmax_vals]

        if hasattr(de_vals, 'tolist'):
            de_vals = de_vals.tolist()
        elif not isinstance(de_vals, list):
            de_vals = [de_vals]

        if hasattr(dsmax_vals, 'tolist'):
            dsmax_vals = dsmax_vals.tolist()
        elif not isinstance(dsmax_vals, list):
            dsmax_vals = [dsmax_vals]

        sid_to_data[sid] = {
            'frames': frames,
            'Fmax': fmax_vals,
            'dE': de_vals,
            'dSmax': dsmax_vals,
        }

    if len(sid_to_data) == 0:
        print(f"[skip] No selected frames for adsorbate '{adsorbate}'", flush=True)
        return (adsorbate, 0, 0)

    print(f"[processing] adsorbate='{adsorbate}'  sids={len(sid_to_data)}  frames={sum(len(v['frames']) for v in sid_to_data.values())}", flush=True)

    writer = LmdbWriter(out_dir=out_dir, entries_per_shard=entries_per_shard, prefix=adsorbate)

    n_traj_ok = 0
    n_frames_extracted = 0

    with tarfile.open(tar_path, "r") as tar:
        # Get all members first
        all_members = [m for m in tar if m.isfile() and m.name.endswith(".extxyz.xz")]

        # Iterate through tar members
        for member in all_members:
            sid = infer_sid_from_member(member.name)
            if sid is None or sid not in sid_to_data:
                continue

            # Read full trajectory
            traj_atoms = read_extxyz_xz_from_tar_member(tar, member)
            if not traj_atoms:
                continue

            n_traj_ok += 1

            # Get the data we need for this sid
            sid_data = sid_to_data[sid]
            frames_to_extract = sid_data['frames']
            fmax_vals = sid_data['Fmax']
            de_vals = sid_data['dE']
            dsmax_vals = sid_data['dSmax']

            # Get initial structure (first frame)
            atoms_init = traj_atoms[0]
    
            # Extract selected frames
            for i, frame_idx in enumerate(frames_to_extract):
                
                atoms = traj_atoms[frame_idx]

                # Get the corresponding Fmax, dE, dSmax values
                Fmax = fmax_vals[i]
                dE = de_vals[i]
                dSmax = dsmax_vals[i]

                # Convert to LMDB format
                data = atoms_to_lmdb_data(
                    atoms=atoms,
                    atoms_init=atoms_init,
                    sid=sid,
                    frame_idx=frame_idx,
                    Fmax=Fmax,
                    dE=dE,
                    dSmax=dSmax,
                )

                writer.add(data)
                n_frames_extracted += 1

    writer.close()

    print(f"[done] adsorbate='{adsorbate}' traj_ok={n_traj_ok} frames_extracted={n_frames_extracted}", flush=True)

    return (adsorbate, n_traj_ok, n_frames_extracted)