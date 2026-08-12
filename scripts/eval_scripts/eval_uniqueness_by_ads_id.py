#!/usr/bin/env python3
"""Compute structural uniqueness per adsorbate id, as in post_eval.py.

Uses the filename→ads_id CSV from list_traj_by_ads.py to group .traj files by
adsorbate id, loads each group, strips the adsorbate atoms (tags == 2), and
runs post_eval's StructureMatcher-based uniqueness on the bare slabs.
The output CSV has one column per adsorbate id and two rows:
the uniqueness ratio and the number of structures it was computed over.

Usage:
    python scripts/uniqueness_by_ads.py \
        --traj_dir for_uma/1_threshold \
        --ads_map traj_by_ads.csv \
        --ads_ids 5 12 \
        --out uniqueness_by_ads.csv
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from gace.utils.post_eval import compute_uniqueness, load_atoms_parallel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, required=True, help="Directory containing .traj files")
    parser.add_argument("--ads_map", type=Path, required=True, help="filename,ads_id CSV from list_traj_by_ads.py")
    parser.add_argument("--ads_ids", type=int, nargs="+", default=None,
                        help="Adsorbate ids to evaluate (default: all resolved ids in the CSV)")
    parser.add_argument("--out", type=Path, default=Path("uniqueness_by_ads.csv"), help="Output CSV path")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Workers for parallel structure loading")
    args = parser.parse_args()

    keep_ads_ids = set(args.ads_ids) if args.ads_ids is not None else None

    # ads_id -> list of traj paths
    files_by_ads = defaultdict(list)
    with open(args.ads_map, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ads_id = int(row["ads_id"])
            except ValueError:
                continue  # no_sid / unreadable / sid_not_in_mapping labels
            if keep_ads_ids is None or ads_id in keep_ads_ids:
                files_by_ads[ads_id].append(str(args.traj_dir / row["filename"]))

    if not files_by_ads:
        raise SystemExit(f"No files in {args.ads_map} match the requested adsorbate ids.")

    results = {}
    for ads_id in sorted(files_by_ads):
        files = files_by_ads[ads_id]
        logging.info(f"ads_id={ads_id}: {len(files)} structures")
        atoms_by_path = load_atoms_parallel(files=files, num_workers=args.num_workers)
        uniqueness_dict = compute_uniqueness(atoms=atoms_by_path.values())
        # compute_uniqueness returns e.g. {'uniqueness (100)': 0.85}; unpack it
        (key, ratio), = uniqueness_dict.items()
        n_structures = int(key.split("(")[1].rstrip(")"))
        results[f"ads_{ads_id}"] = {"uniqueness": ratio, "n_structures": n_structures}
        logging.info(f"ads_id={ads_id}: uniqueness={ratio:.4f} over {n_structures} structures")

    df = pd.DataFrame(results)
    df.to_csv(args.out)
    logging.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
