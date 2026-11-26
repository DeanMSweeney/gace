#!/usr/bin/env python3
"""Copy .traj files for selected adsorbates into <out_dir>/ads_<id>/.

Reads the sid from each traj's atoms.info, resolves the adsorbate id via
oc20_data_mapping.pkl ('random{sid}' keys), and copies the file if its id is
in --ads_ids. The source directory is left untouched.

Usage:
    python scripts/copy_trajs_by_ads_id.py \
        --traj_dir generated_strucs \
        --mapping /path/to/oc20_data_mapping.pkl \
        --ads_ids 0 10 42 54 75 78 \
        --out_dir .
"""

import argparse
import os
import pickle
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from ase.io import read
from tqdm import tqdm


def read_sid(traj_path: Path) -> tuple[Path, int]:
    sid = read(traj_path).info.get("sid")
    if sid is None:
        raise KeyError(f"No sid in atoms.info: {traj_path}")
    return traj_path, int(sid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, required=True, help="Directory containing .traj files")
    parser.add_argument("--mapping", type=Path, required=True, help="Path to oc20_data_mapping.pkl")
    parser.add_argument("--ads_ids", type=int, nargs="+", required=True, help="Adsorbate ids to copy")
    parser.add_argument("--out_dir", type=Path, default=Path("."), help="Root for the ads_<id>/ folders")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    traj_paths = sorted(args.traj_dir.glob("*.traj"))
    if not traj_paths:
        raise FileNotFoundError(f"No .traj files in {args.traj_dir}")

    with Pool(args.num_workers) as pool:
        pairs = list(tqdm(pool.imap(read_sid, traj_paths, chunksize=64), total=len(traj_paths),
                          desc="Reading sids"))

    with open(args.mapping, "rb") as f:
        mapping = pickle.load(f)
    missing = [sid for _, sid in pairs if f"random{sid}" not in mapping]
    if missing:
        raise KeyError(f"{len(missing)} sids not in mapping, e.g. {missing[:5]}")

    keep = set(args.ads_ids)
    copied = Counter()
    for ads_id in keep:
        (args.out_dir / f"ads_{ads_id}").mkdir(parents=True, exist_ok=True)
    for path, sid in tqdm(pairs, desc="Copying"):
        ads_id = int(mapping[f"random{sid}"]["ads_id"])
        if ads_id in keep:
            shutil.copy2(path, args.out_dir / f"ads_{ads_id}" / path.name)
            copied[ads_id] += 1

    print(f"Scanned {len(pairs)} files in {args.traj_dir}")
    for ads_id in sorted(keep):
        print(f"  ads_{ads_id}: {copied[ads_id]}")


if __name__ == "__main__":
    main()
