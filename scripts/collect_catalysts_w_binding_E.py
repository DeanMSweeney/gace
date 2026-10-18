#!/usr/bin/env python
import argparse
import glob
import shutil
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from ase.io.trajectory import Trajectory
from tqdm import tqdm

MATCH_DIR = "matched_traj"


def process_file(traj_path, low, high):
    count = 0
    total = 0
    traj = Trajectory(traj_path)
    for atoms in traj:
        total += 1
        E = atoms.info.get("computed_binding_E")
        if E is not None and low <= E <= high:
            count += 1
    return traj_path, count, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=float, default=-1.00)
    parser.add_argument("--high", type=float, default=1.81)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
    )
    args = parser.parse_args()

    traj_paths = sorted(glob.glob("*_threshold/*.traj"))
    os.makedirs(MATCH_DIR, exist_ok=True)

    worker = partial(process_file, low=args.low, high=args.high)

    count = 0
    total = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for traj_path, c, t in tqdm(
            executor.map(worker, traj_paths),
            total=len(traj_paths),
            desc=f"Scanning ({args.num_workers} workers)",
        ):
            count += c
            total += t
            if c > 0:
                shutil.copy2(traj_path, MATCH_DIR)

    print(f"{count}/{total} samples in [{args.low}, {args.high}]")
