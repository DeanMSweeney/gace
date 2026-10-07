"""Collect convergence info for every slab/adslab run and compute OC20-referenced *H adsorption energies.

    python collect_convergence.py --runs_dir HER_DFT --out_prefix her_dft
Writes <prefix>_runs.csv (one row per run) and <prefix>_ads.csv (one row per system).
"""
import argparse
import csv
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase.io import read
from tqdm import tqdm

E_H = -3.477  # OC20 H reference (eV)

E0_RE = re.compile(r"energy\(sigma->0\) =\s+(-?[\d.]+)")
F_RE = re.compile(r"free  energy   TOTEN  =\s+(-?[\d.]+)")
LOOP_RE = re.compile(r"LOOP\+:")
ELAPSED_RE = re.compile(r"Elapsed time \(sec\):\s+([\d.]+)")


def max_free_force(run_dir, text):
    # Max force norm on unconstrained atoms from the last TOTAL-FORCE block
    try:
        ref = read(run_dir / "POSCAR")
        idx = text.rindex("TOTAL-FORCE")
        lines = text[idx:].splitlines()[2:2 + len(ref)]
        forces = np.array([line.split()[3:6] for line in lines], dtype=float)
    except Exception:
        return None
    free = np.ones(len(ref), dtype=bool)
    for c in ref.constraints:
        free[c.get_indices()] = False
    forces = forces[free]
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
    row["fmax"] = max_free_force(run_dir, text)
    if "reached required accuracy" in text:
        row["status"] = "converged"
    elif elapsed:
        row["status"] = "hit_nsw"  # finished normally without meeting EDIFFG
    else:
        row["status"] = "incomplete"  # running, walltime, or crashed
    return row


def process_system(system_dir):
    slab = parse_run(system_dir / "slab")
    adslab = parse_run(system_dir / "adslab")
    e_ads = None
    if slab["status"] == adslab["status"] == "converged":
        e_ads = round(adslab["e0"] - slab["e0"] - E_H, 4)
    ads = {"system": system_dir.name, "e_slab": slab["e0"], "e_adslab": adslab["e0"], "e_ads": e_ads}
    return [slab, adslab], ads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_prefix", default="convergence")
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()

    system_dirs = sorted(d for d in Path(args.runs_dir).iterdir() if (d / "adslab").is_dir())
    run_rows, ads_rows = [], []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = [pool.submit(process_system, d) for d in system_dirs]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="system"):
            runs, ads = fut.result()
            run_rows += runs
            ads_rows.append(ads)
    run_rows.sort(key=lambda r: (r["system"], r["run"]))
    ads_rows.sort(key=lambda r: r["system"])

    for suffix, rows in (("runs", run_rows), ("ads", ads_rows)):
        with open(f"{args.out_prefix}_{suffix}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    counts = Counter(r["status"] for r in run_rows)
    print("Run status:", dict(counts))
    print(f"Systems with E_ads: {sum(r['e_ads'] is not None for r in ads_rows)}/{len(ads_rows)}")


if __name__ == "__main__":
    main()
