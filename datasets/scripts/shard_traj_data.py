#!/usr/bin/env python3
"""
Stream OC20 per-adsorbate IS2RE/S trajectory tarballs and build sharded metrics.

Assumptions (as per your description):
- Each adsorbate tar contains trajectories organized by system id (sid).
- Each trajectory file is .extxyz.xz inside the tar.
- Frames contain energies and forces readable by ASE.

Output:
<out_root>/<adsorbate>/metrics/part-00000.parquet, part-00001.parquet, ...

Each row corresponds to one frame.
"""

from __future__ import annotations

import argparse
import io
import lzma
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ase.io import read as ase_read

from adsflow.utils.invariance import pbc_aware_delta

from ase.constraints import FixAtoms

SID_RE = re.compile(r"(?:sid[_\-]?|^)(\d+)(?:\D|$)", re.IGNORECASE)


def infer_adsorbate_name(tar_path: Path) -> str:
    """
    Infer an adsorbate folder name from tar filename.
    Examples:
      0.tar -> "0"
      O.tar -> "O"
      per_adsorbate_0.tar -> "per_adsorbate_0"
    If you have a mapping (0->*O), you can swap this function out later.
    """
    return tar_path.stem


def infer_sid_from_member(member_name: str) -> Optional[int]:
    """
    Try to infer sid from the member filename. Works if digits appear in the name.
    If your naming convention is different, modify this.
    """
    base = Path(member_name).name
    m = SID_RE.search(base)
    if m:
        return int(m.group(1))
    # fallback: any digit run
    m2 = re.search(r"(\d+)", base)
    return int(m2.group(1)) if m2 else None


@dataclass
class ShardWriter:
    out_dir: Path
    rows_per_shard: int = 2_000_000
    compression: str = "zstd"

    _rows: List[Dict] = None
    _shard_id: int = 0

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._shard_id = self._find_next_shard_id()

    def _find_next_shard_id(self) -> int:
        """
        Resume-friendly: if part-xxxxx.parquet already exists, continue after max id.
        """
        existing = sorted(self.out_dir.glob("part-*.parquet"))
        if not existing:
            return 0
        max_id = -1
        for p in existing:
            m = re.search(r"part-(\d+)\.parquet$", p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def add_rows(self, rows: List[Dict]) -> None:
        self._rows.extend(rows)
        if len(self._rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        out_path = self.out_dir / f"part-{self._shard_id:05d}.parquet"
        df = pd.DataFrame(self._rows)

        # Convert to Arrow table and write
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_path, compression=self.compression)

        print(f"[write] {out_path}  rows={len(df)}", flush=True)
        self._rows.clear()
        self._shard_id += 1

    def close(self) -> None:
        self.flush()

def get_free_mask(atoms):
    mask = np.ones(len(atoms), dtype=bool)
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            mask[c.index] = 0
    return mask  # True = free atoms


def compute_frame_metrics(traj_atoms: List, sid: int, adsorbate: str) -> List[Dict]:
    """
    Compute per-frame metrics for a single trajectory (one sid).
    Returns one dict per frame.
    """
    nframes = len(traj_atoms)
    if nframes == 0:
        return []

    # compute optimal frame stats (last frame = relaxed structure)
    opt_frame = traj_atoms[-1]
    opt_frame_idx = int(nframes - 1)

    E_opt = opt_frame.get_potential_energy()
    F_opt = opt_frame.get_forces()
    Fmax_opt = (np.linalg.norm(F_opt, axis=1)).max()
    pos_opt = opt_frame.get_positions()

    eps = 1e-12

    # Compute per-frame stats and create one row per frame
    rows = []
    for frame_idx, at in enumerate(traj_atoms):
        # Energy
        E = float(at.get_potential_energy())
        dE = abs(E - E_opt)

        # Max force
        F = at.get_forces()  # (N,3)
        norms = np.linalg.norm(F, axis=1)  # (N,)
        Fmax = float(np.max(norms))
        dFmax = abs(Fmax - Fmax_opt)

        # Position MAE
        positions = at.get_positions()
        mask = get_free_mask(at)
        
        pos_mae = pbc_aware_delta(
            delta=positions - pos_opt,
            cell=at.get_cell(),
            norm=True,
        )
        pos_mae = (np.array(pos_mae) * mask).sum() / mask.sum() 

        row = {
            "adsorbate": adsorbate,
            "sid": int(sid),
            "frame_idx": int(frame_idx),
            "nframes": int(nframes),
            "E": float(E),
            "E_opt": float(E_opt),
            "dE": float(dE),
            "Fmax": float(Fmax),
            "Fmax_opt": float(Fmax_opt),
            "dFmax": float(dFmax),
            "pos_mae": float(pos_mae),
        }
        rows.append(row)

    return rows


def read_extxyz_xz_from_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> List:
    """
    Stream member bytes -> lzma decompress -> ASE read all frames.
    Fully decompresses first to avoid slow seeking in compressed stream.
    """
    f = tar.extractfile(member)
    if f is None:
        return []
    raw = f.read()
    # Decompress .xz completely first (much faster than seeking in compressed stream)
    decompressed = lzma.decompress(raw).decode('utf-8')
    # Now ASE can seek freely in the decompressed StringIO
    atoms_list = ase_read(io.StringIO(decompressed), format="extxyz", index=":")
    # ase_read returns Atoms if single frame, list if multiple
    if atoms_list is None:
        return []
    if isinstance(atoms_list, list):
        return atoms_list
    return [atoms_list]


def iter_tar_members(tar: tarfile.TarFile, suffix: str = ".extxyz.xz") -> Iterable[tarfile.TarInfo]:
    for m in tar:
        if not m.isfile():
            continue
        if m.name.endswith(suffix):
            yield m


def process_single_tar(args_tuple: Tuple[Path, Path, int]) -> Tuple[str, int, int, int]:
    """
    Process a single tar file. Returns (adsorbate, n_members, n_traj_ok, n_traj_skip).
    """
    tar_path, out_root, rows_per_shard = args_tuple

    adsorbate = infer_adsorbate_name(tar_path)
    ads_out = out_root / adsorbate / "metrics"
    writer = ShardWriter(out_dir=ads_out, rows_per_shard=rows_per_shard)

    print(f"adsorbate='{adsorbate}'  metrics_dir='{ads_out}'", flush=True)

    n_members = 0
    n_traj_ok = 0
    n_traj_skip = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in iter_tar_members(tar, suffix=".extxyz.xz"):
            n_members += 1

            sid = infer_sid_from_member(member.name)
            if sid is None:
                n_traj_skip += 1
                continue

            if n_members % 10 == 0:
                print(f"  Processing member {n_members} (sid={sid}, {member.name})", flush=True)

            traj_atoms = read_extxyz_xz_from_tar_member(tar, member)
            frame_rows = compute_frame_metrics(traj_atoms, sid=sid, adsorbate=adsorbate)
            if frame_rows:
                writer.add_rows(frame_rows)
                n_traj_ok += 1
            else:
                n_traj_skip += 1

    writer.close()
    print(
        f"[done] adsorbate='{adsorbate}' members_seen={n_members} traj_ok={n_traj_ok} traj_skipped={n_traj_skip}",
        flush=True,
    )

    return (adsorbate, n_members, n_traj_ok, n_traj_skip)


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
