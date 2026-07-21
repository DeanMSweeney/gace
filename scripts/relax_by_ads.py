"""Relax .traj structures with an MLFF, filtered by adsorbate id, and save energies to CSV.

Uses the filename→ads_id CSV from list_traj_by_ads.py to select structures:
files whose adsorbate id is in the requested set are relaxed with the MLFF.
The output CSV has two columns per adsorbate id — the calculated adsorption
energy and the target binding energy (eV) — with one row per structure.

Outputs are written to <out_dir>/ads_<id>/ so the script can be run once per
adsorbate id without overwriting previous results.

Usage:
    python scripts/relax_by_ads.py \
        --traj_dir for_uma/1_threshold \
        --ads_map traj_by_ads.csv \
        --ads_ids 5
"""

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
from ase.io import read
from fairchem.core import OCPCalculator
from tqdm import tqdm

from gace.utils.relaxation import Silent, relax_w_mlff

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, required=True, help="Directory containing .traj files")
    parser.add_argument("--ads_map", type=Path, required=True, help="filename,ads_id CSV from list_traj_by_ads.py")
    parser.add_argument("--ads_ids", type=int, nargs="+", required=True, help="Adsorbate ids to keep")
    parser.add_argument("--out_dir", type=Path, default=Path("."),
                        help="Root output directory; CSVs go in <out_dir>/ads_<id>/")
    parser.add_argument("--mlff_checkpoint", type=str, default="EquiformerV2-31M-S2EF-OC20-All+MD",
                        help="fairchem model name for OCPCalculator (default matches helm/config.py)")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="BFGS", choices=["BFGS", "LBFGS", "FIRE"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keep_ads_ids = set(args.ads_ids)

    # One folder per run, named by the requested adsorbate id(s)
    run_dir = args.out_dir / ("ads_" + "_".join(str(i) for i in sorted(keep_ads_ids)))
    run_dir.mkdir(parents=True, exist_ok=True)

    # filename -> ads_id for files with a resolved integer id in the requested set
    ads_id_by_file = {}
    with open(args.ads_map, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ads_id = int(row["ads_id"])
            except ValueError:
                continue  # no_sid / unreadable / sid_not_in_mapping labels
            if ads_id in keep_ads_ids:
                ads_id_by_file[row["filename"]] = ads_id

    with Silent():
        calculator = OCPCalculator(
            model_name=args.mlff_checkpoint,
            local_cache="./mlff_checkpoint_cache",
            cpu=False,
            seed=args.seed,
        )

    energies_by_ads = {ads_id: [] for ads_id in sorted(keep_ads_ids)}
    targets_by_ads = {ads_id: [] for ads_id in sorted(keep_ads_ids)}

    logging.info(f"{len(ads_id_by_file)} files in {args.ads_map} match the requested adsorbate ids")

    for filename, ads_id in tqdm(sorted(ads_id_by_file.items()), desc="Relaxing"):
        traj_path = args.traj_dir / filename
        try:
            atoms = read(traj_path)
        except Exception as e:
            logging.warning(f"{filename}: failed to read ({e}), skipping")
            continue

        relaxed_atoms, converged = relax_w_mlff(
            atoms=atoms,
            calculator=calculator,
            fmax=args.fmax,
            steps=args.steps,
            optimizer=args.optimizer,
        )
        if not converged:
            logging.warning(f"{filename}: relaxation did not converge, skipping")
            continue

        energy = relaxed_atoms.get_potential_energy()
        energies_by_ads[ads_id].append(energy)
        # Appended in lockstep with energies so rows align across the two CSVs
        targets_by_ads[ads_id].append(atoms.info.get("target_binding_E", float("nan")))
        logging.info(f"{filename}: ads_id={ads_id} E={energy:.4f} eV target_E={atoms.info.get('target_binding_E', float('nan')):.4f} eV")

    # One CSV with paired columns per adsorbate id; ragged columns padded with NaN
    columns = {}
    for ads_id in sorted(keep_ads_ids):
        columns[f"ads_{ads_id}_adsorption_E"] = energies_by_ads[ads_id]
        columns[f"ads_{ads_id}_target_E"] = targets_by_ads[ads_id]
    df = pd.DataFrame.from_dict(columns, orient="index").transpose()

    out_path = run_dir / "energies.csv"
    df.to_csv(out_path, index=False)
    logging.info(f"Wrote {out_path}: " + ", ".join(f"{c}={df[c].notna().sum()}" for c in df.columns))


if __name__ == "__main__":
    main()
