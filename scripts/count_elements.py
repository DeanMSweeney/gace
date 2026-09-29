#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ase.io import read
from tqdm import tqdm

# Tag value used to mark adsorbate atoms (OC20 / common slab convention:
# 0 = subsurface, 1 = surface, 2 = adsorbate).
ADSORBATE_TAG = 2

# atoms.info key holding the per-structure binding energy.
BINDING_ENERGY_KEY = "computed_binding_E"

# Glob used to discover input directories when none are given on the command line.
DEFAULT_DIR_GLOB = "*_threshold"


def elements_in_structure(atoms):
    """Return the set of unique element symbols in `atoms`, excluding
    adsorbate atoms (tag == ADSORBATE_TAG)."""
    tags = atoms.get_tags()
    symbols = atoms.get_chemical_symbols()
    return {sym for sym, tag in zip(symbols, tags) if tag != ADSORBATE_TAG}


def default_inputs():
    """Return a sorted list of directories matching DEFAULT_DIR_GLOB in the
    current working directory. Used when no input is given on the command line."""
    return sorted(p for p in Path(".").glob(DEFAULT_DIR_GLOB) if p.is_dir())


def gather_traj_files(inputs):
    """Expand a list of paths (files and/or directories) into a sorted list
    of .traj files."""
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.traj")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"Warning: '{path}' does not exist, skipping.", file=sys.stderr)
    return files


def process_file(tf, index):
    """Read one .traj file and return (elem_counts, energy_sum, energy_n,
    n_structures, error_or_None).

    elem_counts: Counter of structures each element appears in.
    energy_sum:  Counter of summed binding energy for structures each
                 element appears in (only structures with a valid
                 BINDING_ENERGY_KEY contribute).
    energy_n:    Counter of how many of those structures had a valid
                 binding energy (denominator for the average).

    Runs in a worker process, so it must not touch shared state.
    """
    try:
        result = read(str(tf), index=index)
    except Exception as exc:
        return Counter(), Counter(), Counter(), 0, str(exc)

    images = result if isinstance(result, list) else [result]
    elem_counts = Counter()
    energy_sum = Counter()
    energy_n = Counter()
    for atoms in images:
        elems = elements_in_structure(atoms)
        elem_counts.update(elems)

        energy = atoms.info.get(BINDING_ENERGY_KEY)
        if energy is not None:
            for el in elems:
                energy_sum[el] += energy
                energy_n[el] += 1

    return elem_counts, energy_sum, energy_n, len(images), None


def main():
    parser = argparse.ArgumentParser(
        description="Count elements across .traj files, excluding adsorbate "
                    "atoms (tag == 2), and report average binding energy "
                    "per element. Each element is counted once per structure.")
    parser.add_argument(
        "input", nargs="*",
        help="One or more .traj files, or directories containing .traj files. "
             "If omitted, every directory matching '%(default)s' in the current "
             "directory is used." % {"default": DEFAULT_DIR_GLOB})
    parser.add_argument(
        "-o", "--output", default="element_counts.csv",
        help="Output CSV path (default: element_counts.csv).")
    parser.add_argument(
        "--index", default="-1",
        help="ASE frame index to read from each trajectory. Default '-1' "
             "(last/relaxed frame). Use ':' to include every frame.")
    parser.add_argument(
        "-j", "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPUs).")
    args = parser.parse_args()

    inputs = args.input
    if not inputs:
        inputs = default_inputs()
        if not inputs:
            sys.exit(f"No input given and no directories matching "
                     f"'{DEFAULT_DIR_GLOB}' found in the current directory.")
        print(f"No input specified; using {len(inputs)} "
              f"'{DEFAULT_DIR_GLOB}' director(ies): "
              f"{', '.join(str(p) for p in inputs)}")

    traj_files = gather_traj_files(inputs)
    if not traj_files:
        sys.exit("No .traj files found.")

    counts = Counter()
    energy_sum = Counter()
    energy_n = Counter()
    n_structures = 0
    n_failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, tf, args.index): tf for tf in traj_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reading trajectories"):
            tf = futures[fut]
            file_counts, file_energy_sum, file_energy_n, n_images, error = fut.result()
            if error is not None:
                print(f"Skipping {tf}: {error}", file=sys.stderr)
                n_failed += 1
                continue
            counts.update(file_counts)
            energy_sum.update(file_energy_sum)
            energy_n.update(file_energy_n)
            n_structures += n_images

    # Sort by count (descending), then alphabetically for ties.
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Element", "Count", "Avg_Binding_Energy_eV", "N_With_Energy"])
        for elem, count in rows:
            n = energy_n[elem]
            avg = f"{energy_sum[elem] / n:.4f}" if n else "N/A"
            writer.writerow([elem, count, avg, n])

    n_missing_energy = sum(1 for elem, _ in rows if energy_n[elem] == 0)

    print(f"Processed {n_structures} structure(s) from "
          f"{len(traj_files) - n_failed} file(s)"
          + (f" ({n_failed} failed)" if n_failed else "") + ".")
    print(f"Wrote {len(rows)} element(s) to {args.output}")
    if n_missing_energy:
        print(f"Warning: {n_missing_energy} element(s) had no structures with "
              f"'{BINDING_ENERGY_KEY}' set.", file=sys.stderr)


if __name__ == "__main__":
    main()
