"""Collect convergence info for every slab/adslab run and compute OC20-referenced *H adsorption energies.

    python collect_convergence.py --runs_dir HER_DFT --out_prefix her_dft
Writes <prefix>_runs.csv (one row per run) and <prefix>_ads.csv (one row per system).
With --adslab_only, only adslab runs are parsed and only <prefix>_runs.csv is written.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read
from tqdm import tqdm

E0_RE = re.compile(r"energy\(sigma->0\) =\s+(-?[\d.]+)")
F_RE = re.compile(r"free  energy   TOTEN  =\s+(-?[\d.]+)")
LOOP_RE = re.compile(r"LOOP\+:")
ELAPSED_RE = re.compile(r"Elapsed time \(sec\):\s+([\d.]+)")


def max_free_force(run_dir):
    # Max force norm on unconstrained atoms from the last OUTCAR frame
    try:
        atoms = read(run_dir / "OUTCAR", index=-1)
        ref = read(run_dir / "POSCAR")
    except Exception:
        return None
    free = np.ones(len(atoms), dtype=bool)
    for c in ref.constraints:
        free[c.get_indices()] = False
    forces = atoms.get_forces()[free]
    return round(float(np.linalg.norm(forces, axis=1).max()), 4) if len(forces) else None


def parse_run(run_dir):
    row = {"system": run_dir.parent.name, "run": run_dir.name, "status": "not_started",
           "n_ionic_steps": 0, "e0": None, "free_energy": None, "fmax": None, "elapsed_s": None}
    outcar = run_dir / "OUTCAR"
    if not outcar.exists():
        return row
    text = outcar.read_text(errors="ignore")
    e0 = E0_RE.findall(text)
    f = F_RE.findall(text)
    elapsed = ELAPSED_RE.search(text)
    row["n_ionic_steps"] = len(LOOP_RE.findall(text))
    row["e0"] = float(e0[-1]) if e0 else None
    row["free_energy"] = float(f[-1]) if f else None
    row["elapsed_s"] = float(elapsed.group(1)) if elapsed else None
    row["fmax"] = max_free_force(run_dir)
    if "reached required accuracy" in text:
        row["status"] = "converged"
    elif elapsed:
        row["status"] = "hit_nsw"  # finished normally without meeting EDIFFG
    else:
        row["status"] = "incomplete"  # running, walltime, or crashed
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_prefix", default="convergence")
    parser.add_argument("--e_ref", type=float,
                        help="Gas-phase reference energy of the adsorbate (eV)")
    parser.add_argument("--adslab_only", action="store_true",
                        help="Only collect adslab runs; skip slabs and E_ads")
    args = parser.parse_args()

    system_dirs = sorted(d for d in Path(args.runs_dir).iterdir() if (d / "adslab").is_dir())
    run_rows, ads_rows = [], []
    for system_dir in tqdm(system_dirs, unit="system"):
        adslab = parse_run(system_dir / "adslab")
        if args.adslab_only:
            run_rows.append(adslab)
            continue
        slab = parse_run(system_dir / "slab")
        run_rows += [slab, adslab]

        e_ads = None
        if slab["status"] == adslab["status"] == "converged":
            e_ads = round(adslab["e0"] - slab["e0"] - args.e_ref, 4)
        ads_rows.append({"system": system_dir.name,
                         "e_slab": slab["e0"], "e_adslab": adslab["e0"], "e_ads": e_ads})

    outputs = [("runs", run_rows)] if args.adslab_only else [("runs", run_rows), ("ads", ads_rows)]
    for suffix, rows in outputs:
        with open(f"{args.out_prefix}_{suffix}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    counts = Counter(r["status"] for r in run_rows)
    print("Run status:", dict(counts))
    if not args.adslab_only:
        print(f"Systems with E_ads: {sum(r['e_ads'] is not None for r in ads_rows)}/{len(ads_rows)}")


if __name__ == "__main__":
    main()
