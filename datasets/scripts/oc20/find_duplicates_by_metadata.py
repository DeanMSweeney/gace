#!/usr/bin/env python3
"""Group .traj files by catalyst and report duplicates.

Reads the sid from each traj's atoms.info, resolves the catalyst identity via
oc20_mapping.pkl, and writes a CSV with columns filename,sid,ads_id,cat_id.
Files that can't be resolved get a string label (no_sid, sid_not_in_mapping,
unreadable) in the cat_id column.

OC20 has no explicit catalyst id, so it is reconstructed as the tuple
(bulk_id, miller_index, shift, top), which uniquely specifies the surface.
bulk_id alone is NOT enough -- same bulk, different Miller index is a
different catalyst. shift is a float and is rounded before comparison.

Pass --group_by cat_ads to group by (catalyst, adsorbate) instead, which is
what you want if the traj_dir mixes adsorbates.

Usage:
    python scripts/list_traj_by_catalyst.py \
        --traj_dir for_uma/1_threshold \
        --mapping /path/to/oc20_mapping.pkl \
        --out traj_by_catalyst.csv
"""

import argparse
import csv
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ase.io import read
from tqdm import tqdm

SHIFT_DECIMALS = 4


def lookup_entry(mapping: dict, sid: int):
    """OC20 mapping keys are usually 'random{sid}'; fall back to raw int/str keys."""
    for key in (f"random{sid}", sid, str(sid)):
        if key in mapping:
            return mapping[key]
    return None


def catalyst_id(entry: dict):
    """Tuple uniquely specifying the surface: bulk, facet, termination, side."""
    return (
        entry["bulk_id"],
        tuple(entry["miller_index"]),
        round(float(entry["shift"]), SHIFT_DECIMALS),
        bool(entry["top"]),
    )


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
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: <traj_dir>/traj_by_catalyst.csv)")
    parser.add_argument("--group_by", choices=["cat", "cat_ads"], default="cat",
                        help="Group by catalyst alone, or by (catalyst, adsorbate)")
    parser.add_argument("--dup_list", type=Path, default=None,
                        help="Output path for the list of redundant filenames, one kept per group "
                             "(default: <traj_dir>/duplicates.txt)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    if args.out is None:
        args.out = args.traj_dir / "traj_by_catalyst.csv"
    if args.dup_list is None:
        args.dup_list = args.traj_dir / "duplicates.txt"

    with open(args.mapping, "rb") as f:
        mapping = pickle.load(f)

    traj_paths = sorted(args.traj_dir.glob("*.traj"))

    # Read trajs in parallel; mapping lookup stays in the main process
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        results = list(tqdm(pool.map(read_sid, traj_paths), total=len(traj_paths), desc="Reading trajs"))

    rows = []
    for name, sid, error in results:
        if error is not None:
            rows.append((name, "", "", "unreadable"))
            continue
        if sid is None:
            rows.append((name, "", "", "no_sid"))
            continue

        entry = lookup_entry(mapping=mapping, sid=int(sid))
        if entry is None:
            rows.append((name, sid, "", "sid_not_in_mapping"))
            continue

        cid = catalyst_id(entry)
        ads_id = entry["ads_id"]
        key = (cid, ads_id) if args.group_by == "cat_ads" else cid
        rows.append((name, sid, ads_id, repr(key)))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "sid", "ads_id", "cat_id"])
        writer.writerows(rows)

    unresolved = {"unreadable", "no_sid", "sid_not_in_mapping"}
    keys = [r[3] for r in rows if r[3] not in unresolved]
    counts = Counter(keys)

    sids_by_group = defaultdict(list)
    files_by_group = defaultdict(list)
    for name, sid, ads_id, key in rows:
        if key not in unresolved:
            sids_by_group[key].append(sid)
            files_by_group[key].append(name)

    # Keep the first file in each group (sorted, so the choice is deterministic
    # and reproducible across runs); everything after it is redundant.
    redundant = []
    for key, names in files_by_group.items():
        redundant.extend(sorted(names)[1:])
    redundant.sort()

    with open(args.dup_list, "w") as f:
        f.write("\n".join(redundant))
        if redundant:
            f.write("\n")

    n_files = len(rows)
    n_resolved = len(keys)
    n_groups = len(counts)
    n_dup = n_resolved - n_groups
    largest = max(counts.values()) if counts else 0
    n_repeated = sum(1 for c in counts.values() if c > 1)

    print(f"Wrote {args.out}")
    print(f"  files:               {n_files}")
    print(f"  resolved:            {n_resolved}")
    for label in sorted(unresolved):
        n = sum(1 for r in rows if r[3] == label)
        if n:
            print(f"    {label}: {n}")
    print(f"  distinct groups:     {n_groups}  (group_by={args.group_by})")
    print(f"  redundant files:     {n_dup}  ({100.0 * n_dup / n_resolved:.2f}% of resolved)" if n_resolved else "")
    print(f"  groups with >1 file: {n_repeated}")
    print(f"  largest group:       {largest}")
    print(f"\nWrote {args.dup_list}: {len(redundant)} redundant files (one kept per group)")
    print(f"  move them with:  mkdir -p dupes && xargs -a {args.dup_list} -d '\\n' -I{{}} "
          f"mv {args.traj_dir}/{{}} dupes/")

    if n_repeated:
        print("\n  top repeated groups:")
        for key, c in counts.most_common(10):
            if c < 2:
                break
            sids = ", ".join(str(s) for s in sids_by_group[key])
            print(f"    {c:>5}  {key}")
            print(f"           sids: {sids}")


if __name__ == "__main__":
    main()
