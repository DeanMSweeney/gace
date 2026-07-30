"""Map .traj filenames to adsorbate ids.

Reads the sid from each traj's atoms.info, resolves the adsorbate id via
oc20_mapping.pkl, and writes a CSV with columns filename,ads_id. Files that
can't be resolved get a string label (no_sid, sid_not_in_mapping, unreadable)
in the ads_id column.

Usage:
    python scripts/list_traj_by_ads.py \
        --traj_dir for_uma/1_threshold \
        --mapping /path/to/oc20_mapping.pkl \
        --out traj_by_ads.csv
"""

import argparse
import csv
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ase.io import read
from tqdm import tqdm


def lookup_ads_id(mapping: dict, sid: int):
    """OC20 mapping keys are usually 'random{sid}'; fall back to raw int/str keys."""
    for key in (f"random{sid}", sid, str(sid)):
        if key in mapping:
            return mapping[key]["ads_id"]
    return None


def read_sid(traj_path: Path):
    """Worker: read one traj and return (filename, sid, error). Runs in a subprocess."""
    try:
        atoms = read(traj_path)
    except Exception as e:
        return traj_path.name, None, str(e)
    return traj_path.name, atoms.info.get("sid"), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, required=True, help="Directory containing .traj files")
    parser.add_argument("--mapping", type=Path, required=True, help="Path to oc20_mapping.pkl")
    parser.add_argument("--out", type=Path, default=Path("traj_by_ads.csv"), help="Output CSV path")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.mapping, "rb") as f:
        mapping = pickle.load(f)

    traj_paths = sorted(args.traj_dir.glob("*.traj"))

    # Read trajs in parallel; mapping lookup stays in the main process
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(read_sid, traj_paths), total=len(traj_paths), desc="Reading trajs"))

    rows = []
    for name, sid, error in results:
        if error is not None:
            rows.append((name, "unreadable"))
        elif sid is None:
            rows.append((name, "no_sid"))
        else:
            ads_id = lookup_ads_id(mapping=mapping, sid=int(sid))
            rows.append((name, ads_id if ads_id is not None else "sid_not_in_mapping"))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ads_id"])
        writer.writerows(rows)

    n_resolved = sum(1 for _, ads_id in rows if isinstance(ads_id, int))
    print(f"Wrote {args.out}: {n_resolved}/{len(rows)} files resolved to an adsorbate id")


if __name__ == "__main__":
    main()
