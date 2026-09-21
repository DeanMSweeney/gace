#!/usr/bin/env python
"""
Two steps, run on different machines if needed.

1. Where the trajectories live (no network needed):
    python build_mp_cache.py --traj_dir run1/ run2/ --num_workers 16
   -> writes chemsys.txt

2. Where there's network and mp_api:
    export MP_API_KEY=...
    python build_mp_cache.py --num_workers 4
   -> writes .ehull_cache/

Then copy .ehull_cache/ next to ehull_from_traj.py.
"""
import argparse
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


# --------------------------------------------------------------------------
# step 1: trajectories -> chemsys.txt  (CPU/IO bound -> processes)
# --------------------------------------------------------------------------

def _chemsys_of(path):
    from ase.io import read
    prim = read(path, index=-1).info.get("primitive_cell")
    if prim is None:
        return path.name, None
    syms = (prim.get_chemical_symbols() if hasattr(prim, "get_chemical_symbols")
            else [str(s.specie) for s in prim])
    return path.name, frozenset(syms)


def chemsys_from_trajs(traj_dirs, num_workers):
    files = sorted(f for d in traj_dirs for f in d.glob("*.traj"))
    systems, skipped = set(), []
    with ProcessPoolExecutor(num_workers) as ex:
        for name, s in tqdm(ex.map(_chemsys_of, files, chunksize=32),
                            total=len(files), desc="scanning trajs"):
            if s is None:
                skipped.append(name)
            else:
                systems.add(s)
    if skipped:
        print(f"skipped {len(skipped)} files with no atoms.info['primitive_cell'], "
              f"e.g. {skipped[:3]}")
    return systems


# --------------------------------------------------------------------------
# step 2: chemsys.txt -> cache  (network bound -> threads)
# --------------------------------------------------------------------------

def read_chemsys(path):
    systems = set()
    for line in path.read_text().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            systems.add(frozenset(line.replace("-", " ").split()))
    return systems


_local = threading.local()


def _rester():
    # one client per thread; sharing a requests session across threads is unsafe
    if not hasattr(_local, "mpr"):
        from mp_api.client import MPRester
        _local.mpr = MPRester()
    return _local.mpr


def fetch(elements, cache_dir):
    path = cache_dir / f"{'-'.join(sorted(elements))}.pkl"
    # pin to the GGA/GGA+U hull: MP2020 corrections target that scale,
    # not MP's newer joint GGA/GGA+U/r2SCAN default
    entries = _rester().get_entries_in_chemsys(
        sorted(elements), additional_criteria={"thermo_types": ["GGA_GGA+U"]})
    path.write_bytes(pickle.dumps(entries))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", type=Path, nargs="+",
                   help="scan these directories and write --chemsys_file")
    p.add_argument("--chemsys_file", type=Path, default=Path("chemsys.txt"))
    p.add_argument("--cache_dir", type=Path, default=Path(".ehull_cache"))
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    if args.traj_dir:
        systems = chemsys_from_trajs(args.traj_dir, args.num_workers)
        args.chemsys_file.write_text(
            "\n".join(sorted("-".join(sorted(s)) for s in systems)) + "\n")
        print(f"wrote {len(systems)} systems to {args.chemsys_file}")
        return

    systems = read_chemsys(args.chemsys_file)
    elements = set().union(*systems)
    systems |= {frozenset([el]) for el in elements}   # unaries for elemental refs

    args.cache_dir.mkdir(exist_ok=True)
    todo = [s for s in systems
            if not (args.cache_dir / f"{'-'.join(sorted(s))}.pkl").exists()]
    print(f"{len(systems)} systems, {len(systems) - len(todo)} already cached, "
          f"{len(todo)} to fetch")

    failed = []
    with ThreadPoolExecutor(args.num_workers) as ex:
        futures = {ex.submit(fetch, s, args.cache_dir): s for s in todo}
        for fut in tqdm(futures, desc="fetching MP"):
            try:
                fut.result()
            except Exception as exc:
                failed.append(("-".join(sorted(futures[fut])), exc))
    if failed:
        print(f"{len(failed)} failed (rerun to retry):")
        for name, exc in failed[:10]:
            print(f"  {name}: {exc}")


if __name__ == "__main__":
    main()
