"""Relax .traj structures with an MLFF, filtered by adsorbate id, and save energies to CSV.

For each .traj file: read sid from atoms.info, look up the adsorbate id in
oc20_mapping.pkl, and if it is in the requested set, relax with the MLFF and
record the final potential energy. The output CSV has one column per adsorbate
id, rows are the energies (eV) of the structures with that adsorbate.

Usage:
    python scripts/relax_by_ads.py \
        --traj_dir for_uma/1_threshold \
        --mapping /path/to/oc20_mapping.pkl \
        --ads_ids 5 12 74 \
        --out energies_by_ads.csv
"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from ase.io import read
from fairchem.core import OCPCalculator
from tqdm import tqdm

from gace.utils.relaxation import Silent, relax_w_mlff

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    parser.add_argument("--ads_ids", type=int, nargs="+", required=True, help="Adsorbate ids to keep")
    parser.add_argument("--out", type=Path, default=Path("energies_by_ads.csv"), help="Output CSV path")
    parser.add_argument("--target_out", type=Path, default=Path("target_binding_E_by_ads.csv"),
                        help="Output CSV path for target binding energies")
    parser.add_argument("--mlff_checkpoint", type=str, default="EquiformerV2-31M-S2EF-OC20-All+MD",
                        help="fairchem model name for OCPCalculator (default matches helm/config.py)")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="BFGS", choices=["BFGS", "LBFGS", "FIRE"])
    parser.add_argument("--gpu", action="store_true", help="Run the MLFF on GPU")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.mapping, "rb") as f:
        mapping = pickle.load(f)

    keep_ads_ids = set(args.ads_ids)

    with Silent():
        calculator = OCPCalculator(
            model_name=args.mlff_checkpoint,
            local_cache="./mlff_checkpoint_cache",
            cpu=not args.gpu,
            seed=args.seed,
        )

    energies_by_ads = {ads_id: [] for ads_id in sorted(keep_ads_ids)}
    targets_by_ads = {ads_id: [] for ads_id in sorted(keep_ads_ids)}

    traj_files = sorted(args.traj_dir.glob("*.traj"))
    logging.info(f"Found {len(traj_files)} .traj files in {args.traj_dir}")

    for traj_path in tqdm(traj_files, desc="Relaxing"):
        try:
            atoms = read(traj_path)
        except Exception as e:
            logging.warning(f"{traj_path.name}: failed to read ({e}), skipping")
            continue

        sid = atoms.info.get("sid")
        if sid is None:
            logging.warning(f"{traj_path.name}: no sid in atoms.info, skipping")
            continue

        ads_id = lookup_ads_id(mapping=mapping, sid=int(sid))
        if ads_id is None:
            logging.warning(f"{traj_path.name}: sid {sid} not in mapping, skipping")
            continue
        if ads_id not in keep_ads_ids:
            continue

        relaxed_atoms, converged = relax_w_mlff(
            atoms=atoms,
            calculator=calculator,
            fmax=args.fmax,
            steps=args.steps,
            optimizer=args.optimizer,
        )
        if not converged:
            logging.warning(f"{traj_path.name}: relaxation did not converge, skipping")
            continue

        energy = relaxed_atoms.get_potential_energy()
        energies_by_ads[ads_id].append(energy)
        # Appended in lockstep with energies so rows align across the two CSVs
        targets_by_ads[ads_id].append(atoms.info.get("target_binding_E", float("nan")))
        logging.info(f"{traj_path.name}: sid={sid} ads_id={ads_id} E={energy:.4f} eV")

    def write_wide_csv(values_by_ads: dict, out_path: Path):
        # Pad ragged columns with NaN so every adsorbate id gets its own column
        df = pd.DataFrame.from_dict(values_by_ads, orient="index").transpose()
        df.columns = [f"ads_{ads_id}" for ads_id in df.columns]
        df.to_csv(out_path, index=False)
        logging.info(f"Wrote {out_path}: " + ", ".join(f"{c}={df[c].notna().sum()}" for c in df.columns))

    write_wide_csv(values_by_ads=energies_by_ads, out_path=args.out)
    write_wide_csv(values_by_ads=targets_by_ads, out_path=args.target_out)


if __name__ == "__main__":
    main()
