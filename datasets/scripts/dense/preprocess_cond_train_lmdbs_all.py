"""
Creates LMDB files containing raw numpy dict entries for all trajectory final
positions per system, tagged with energy relative to the lowest-energy trajectory.

For each system ID:
  - Both the initial (index=0) and final (index=-1) frames of every trajectory are read
  - Energies are computed from final frames and shifted so the minimum = 0.0 (ΔE)
  - Initial frames are derange-shuffled so that pos and pos_relaxed always come
    from *different* trajectories — the model learns to map any starting placement
    to a low-energy final placement conditioned on ΔE, not a specific path
"""

import argparse
import multiprocessing as mp
import os
import pickle
import glob

import lmdb
import numpy as np
import ase.io
from ase.constraints import FixAtoms


def _get_fixed_mask(atoms):
    """Return an int8 array: 1 where atom is fixed (FixAtoms constraint)."""
    mask = np.zeros(len(atoms), dtype=np.int8)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[constraint.index] = 1
    return mask


def deranged_shuffle(n):
    """
    Return a permutation of range(n) with no fixed points (derangement).
    Uses rejection sampling — terminates in O(1) expected attempts since
    ~37% of random permutations are already derangements.
    Falls back to a cyclic shift if n == 1 (derangement impossible).
    """
    if n == 1:
        return [0]  # can't derange a single element
    indices = list(range(n))
    while True:
        np.random.shuffle(indices)
        if all(indices[i] != i for i in range(n)):
            return indices


def write_images_to_lmdb(mp_arg):
    db_path, samples, sampled_ids, idx, pid, args, trajs_dir, tags_map = mp_arg

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    print(db_path)

    for sys_id in samples:
        sid = sys_id.strip()
        traj_dir = os.path.join(trajs_dir, sid)
        traj_files = glob.glob(traj_dir + "/*[!surface].traj")

        if not traj_files:
            print(f"Warning: no traj files found for {sid}, skipping.")
            continue

        # Read both initial (index=0) and final (index=-1) frames per trajectory
        initial_frames = [ase.io.read(p, index=0)  for p in traj_files]
        final_frames   = [ase.io.read(p, index=-1) for p in traj_files]

        # Compute relative energies from final frames
        energies = np.array(
            [atoms.get_potential_energy() for atoms in final_frames],
            dtype=np.float32,
        )
        minE_idx = np.argmin(energies)
        energies -= energies[minE_idx]
        assert energies[minE_idx] == 0.0

        # Derange-shuffle initial frames so pos and pos_relaxed always come
        # from different trajectories (no same-trajectory leakage)
        n = len(traj_files)
        shuffled_init_indices = deranged_shuffle(n)

        # Store every entry: minimum-energy frame first (fid=-1), rest by index
        ordered_indices = [minE_idx] + [i for i in range(n) if i != minE_idx]

        for fid_label, i in enumerate(ordered_indices):
            final_atoms   = final_frames[i]
            initial_atoms = initial_frames[shuffled_init_indices[i]]

            data = {
                # pos: initial placement from a *different* trajectory (deranged)
                "pos":            initial_atoms.get_positions().astype(np.float32),
                # pos_relaxed: the DFT-relaxed final frame
                "pos_relaxed":    final_atoms.get_positions().astype(np.float32),
                "cell":           np.array(final_atoms.get_cell()).astype(np.float32),
                "atomic_numbers": final_atoms.get_atomic_numbers().astype(np.int64),
                "natoms":         len(final_atoms),
                "fixed":          _get_fixed_mask(final_atoms),
                "tags":           np.array(tags_map[sid], dtype=np.int8),
                "sid":            sid,
                "fid":            -1 if i == minE_idx else i,
                "y":              float(energies[i]),  # relative energy (ΔE)
            }

            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
            idx += 1

    # Save count of objects in lmdb
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build LMDB shards (raw dict format) from all OC20 dense trajectory "
            "final frames, tagged with relative energy (ΔE vs. lowest-energy frame)."
        )
    )
    parser.add_argument(
        "--txt-path",
        required=True,
        help="Path to the unique_train_system_id.txt file",
    )
    parser.add_argument(
        "--trajs-dir",
        required=True,
        help="Root directory containing per-system trajectory subdirectories",
    )
    parser.add_argument(
        "--tags-path",
        required=True,
        help="Path to the oc20dense_tags.pkl file",
    )
    parser.add_argument(
        "--mapping-path",
        required=True,
        help="Path to the oc20dense_mapping.pkl file",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Directory to save output LMDB shards. Will be created if it doesn't exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes / output LMDB shards (default: 1)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which split to preprocess: 'train' uses system_ids[200:], 'val' uses system_ids[:200] (default: train)",
    )
    parser.add_argument(
        "--chunk", default=1, type=int, help="Chunk of inputs to preprocess"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run sequentially (no multiprocessing) for easy debugging",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with open(args.tags_path, "rb") as h:
        tags_map = pickle.load(h)

    with open(args.mapping_path, "rb") as f:
        oc20_dense_mapping = pickle.load(f)

    # Read and shuffle system IDs with a fixed seed for reproducibility
    with open(args.txt_path, "r") as f:
        system_ids = f.readlines()

    np.random.seed(42)
    np.random.shuffle(system_ids)

    # Slice based on split: first 200 = val, rest = train
    if args.split == "val":
        system_ids = system_ids[:200]
    else:
        system_ids = system_ids[200:]
    print("### Found %d trajectories for split='%s'" % (len(system_ids), args.split))

    # Create output directory if needed
    os.makedirs(args.out_path, exist_ok=True)

    # One LMDB shard per worker
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Split system IDs across workers
    chunked_traj_dirs = np.array_split(system_ids, args.num_workers)

    sampled_ids = [[]] * args.num_workers
    idx = [0] * args.num_workers

    mp_args = [
        (
            db_paths[i],
            chunked_traj_dirs[i],
            sampled_ids[i],
            idx[i],
            i,
            args,
            args.trajs_dir,
            tags_map,
        )
        for i in range(args.num_workers)
    ]

    if args.debug:
        op = [write_images_to_lmdb(a) for a in mp_args]
        op = list(zip(*op))
    else:
        pool = mp.Pool(args.num_workers)
        op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))