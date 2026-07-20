"""List .traj filenames grouped by adsorbate id.

Reads the sid from each traj's atoms.info, resolves the adsorbate id via
oc20_mapping.pkl, and writes a .txt with one section per adsorbate id.

Usage:
    python scripts/list_traj_by_ads.py \
        --traj_dir for_uma/1_threshold \
        --mapping /path/to/oc20_mapping.pkl \
        --out traj_by_ads.txt
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

from ase.io import read


def lookup_ads_id(mapping: dict, sid: int):
    """OC20 mapping keys are usually 'random{sid}'; fall back to raw int/str keys."""
    for key in (f"random{sid}", sid, str(sid)):
        if key in mapping:
            return mapping[key]["ads_id"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, required=True, help="Directory containing .traj files")
    parser.add_argument("--mapping", type=Path, required=True, help="Path to oc20_mapping.pkl")
    parser.add_argument("--out", type=Path, default=Path("traj_by_ads.txt"), help="Output txt path")
    args = parser.parse_args()

    with open(args.mapping, "rb") as f:
        mapping = pickle.load(f)

    files_by_ads = defaultdict(list)
    for traj_path in sorted(args.traj_dir.glob("*.traj")):
        try:
            atoms = read(traj_path)
        except Exception as e:
            files_by_ads["unreadable"].append(f"{traj_path.name}  ({e})")
            continue

        sid = atoms.info.get("sid")
        if sid is None:
            files_by_ads["no_sid"].append(traj_path.name)
            continue

        ads_id = lookup_ads_id(mapping=mapping, sid=int(sid))
        if ads_id is None:
            files_by_ads["sid_not_in_mapping"].append(traj_path.name)
            continue

        files_by_ads[ads_id].append(traj_path.name)

    with open(args.out, "w") as f:
        # Integer ads ids first (sorted), then any string buckets (no_sid, etc.)
        keys = sorted([k for k in files_by_ads if isinstance(k, int)])
        keys += sorted([k for k in files_by_ads if not isinstance(k, int)])
        for key in keys:
            header = f"ads_{key}" if isinstance(key, int) else key
            f.write(f"{header} ({len(files_by_ads[key])} files)\n")
            for name in files_by_ads[key]:
                f.write(f"    {name}\n")
            f.write("\n")

    print(f"Wrote {args.out} ({sum(len(v) for v in files_by_ads.values())} files across {len(keys)} groups)")


if __name__ == "__main__":
    main()
