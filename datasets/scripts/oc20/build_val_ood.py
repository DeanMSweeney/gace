"""
Build a forge-eval LMDB from the OC20 IS2RE val_ood split.

For each val_ood entry we keep only the adsorbate conditioning signals —
initial adsorbate positions (tags == 2), atomic numbers, binding energy
(y_relaxed), and sid. Adsorbate positions are centered by COM and given a
random SO(3) rotation (inference-time sample() only re-centers, never
rotates). All catalyst fields are dummy placeholders, reinitialised from
the prior inside sample().

Entries are written as CatFlow-format dicts so LazyLmdbDataset loads them
through catflow_dict_to_pyg_data, exactly like val_id's
dataset_processed.lmdb.

Usage
-----
    python build_val_ood_forge_lmdb.py --src /path/to/val_ood.lmdb \
        --dst /datasets/factorized/val_ood/dataset_processed.lmdb \
        [--limit N] [--seed S] [--num-prim-atoms 4]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Numpy compatibility shim (data pickled with numpy>=1.25, redirects _core -> core)
import numpy, numpy.core
if not hasattr(numpy, '_core'):
    numpy._core = numpy.core
    import pkgutil
    for _, modname, _ in pkgutil.walk_packages(numpy.core.__path__, prefix='numpy.core.', onerror=lambda x: None):
        alias = modname.replace('numpy.core', 'numpy._core')
        if modname in sys.modules:
            sys.modules[alias] = sys.modules[modname]

import lmdb
import numpy as np
import torch
from ase import Atoms
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gace.utils.utils import center_and_random_so3_rotation


def _to_np(x):
    """Convert tensor / numpy array / list to plain numpy array."""
    try:
        return x.cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def build_entry(data, num_prim_atoms: int) -> dict | None:
    """Build a CatFlow-format dict from a raw OC20 IS2RE entry, or None to skip."""
    try:
        tags = _to_np(data.tags).astype(int).flatten()
        pos = _to_np(data.pos).astype(np.float32)
        numbers = _to_np(data.atomic_numbers).astype(np.int64).flatten()
        y_relaxed = float(data.y_relaxed)
        sid = int(data.sid)
    except AttributeError:
        return None

    ads_mask = tags == 2
    if not ads_mask.any():
        return None

    ads_pos = torch.from_numpy(pos[ads_mask])
    n_ads = int(ads_mask.sum())
    ads_centered = center_and_random_so3_rotation(
        ref_pos=ads_pos,
        ads_batch=torch.zeros(n_ads, dtype=torch.long),
        B=1,
    ).numpy()

    return {
        # Adsorbate conditioning (real values)
        "ads_atomic_numbers": numbers[ads_mask],
        "ads_pos": ads_centered,
        "ref_ads_pos": ads_centered,
        "ads_rel_pos": ads_centered,
        "ads_com": np.zeros(3, dtype=np.float32),
        "ref_energy": y_relaxed,
        "sid": sid,
        # Dummy catalyst fields — sample() reinitialises them from the prior
        "primitive_slab": Atoms(
            numbers=[1] * num_prim_atoms,
            positions=np.zeros((num_prim_atoms, 3)),
            cell=np.eye(3),
            pbc=True,
        ),
        "frac_coords": np.zeros((num_prim_atoms, 3), dtype=np.float32),
        "lengths_scaled": np.zeros(3, dtype=np.float32),
        "angles_radians": np.zeros(3, dtype=np.float32),
        "supercell_matrix": np.eye(3, dtype=np.int64),
        "n_slab": 0,
        "n_vac": 0,
        "bind_ads_atom": None,
    }


def run(src: Path, dst: Path, limit: int | None, seed: int, num_prim_atoms: int) -> None:
    torch.manual_seed(seed)

    src_env = lmdb.open(
        str(src), subdir=False, readonly=True,
        lock=False, readahead=False, meminit=False, max_readers=1,
    )
    with src_env.begin() as txn:
        length_entry = txn.get(b"length")
        total = pickle.loads(length_entry) if length_entry is not None else src_env.stat()["entries"]

    n_scan = total if limit is None else min(limit, total)
    print(f"Source  : {src}  ({total} total samples)")
    print(f"Output  : {dst}")
    print(f"Process : {n_scan} samples  |  num_prim_atoms: {num_prim_atoms}  |  seed: {seed}")

    entries: list[bytes] = []
    n_skipped = 0
    with src_env.begin() as txn:
        for idx in tqdm(range(n_scan), desc="Building entries"):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                n_skipped += 1
                continue
            entry = build_entry(data=pickle.loads(raw), num_prim_atoms=num_prim_atoms)
            if entry is None:
                n_skipped += 1
                continue
            entries.append(pickle.dumps(entry, protocol=4))
    src_env.close()

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst_env = lmdb.open(str(dst), subdir=False, readonly=False, map_size=2**32, meminit=False)
    with dst_env.begin(write=True) as txn:
        for i, data in enumerate(tqdm(entries, desc="Writing LMDB")):
            txn.put(f"{i}".encode("ascii"), data)
        txn.put(b"length", pickle.dumps(len(entries), protocol=4))
    dst_env.close()

    print(f"Done. Wrote {len(entries)} samples → {dst}  (skipped {n_skipped})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a forge-eval CatFlow-format LMDB from OC20 val_ood adsorbates."
    )
    parser.add_argument("--src", required=True, help="Path to OC20 val_ood.lmdb")
    parser.add_argument("--dst", default="/datasets/factorized/val_ood/dataset_processed.lmdb",
                        help="Output LMDB file path")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random SO(3) rotations")
    parser.add_argument("--num-prim-atoms", type=int, default=4,
                        help="Number of dummy primitive catalyst atoms")
    args = parser.parse_args()

    run(
        src=Path(args.src),
        dst=Path(args.dst),
        limit=args.limit,
        seed=args.seed,
        num_prim_atoms=args.num_prim_atoms,
    )


if __name__ == "__main__":
    main()
