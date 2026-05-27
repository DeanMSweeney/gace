"""Relax a random subset of generated structures with an MLFF and evaluate binding energies.

Usage:
    python scripts/relax_gen_strucs.py \
        --gen_strucs ./gen_strucs \
        --rlxd_strucs ./rlxd_strucs \
        --n_strucs 1000
"""

import argparse
import csv
import glob
import logging
import os
import random

import ase.io
from fairchem.core import OCPCalculator
from tqdm import tqdm

from gace.utils.relaxation import Silent, _fmt, relax_w_mlff

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Relax forge-generated structures with MLFF.")
    p.add_argument("--gen_strucs", required=True, help="Directory containing generated .vasp files and binding_energies.csv")
    p.add_argument("--rlxd_strucs", default="./rlxd_strucs", help="Output directory for relaxed structures")
    p.add_argument("--mlff_checkpoint", default="EquiformerV2-31M-S2EF-OC20-All+MD", help="fairchem model name")
    p.add_argument("--energy_cutoff", type=float, default=10.0, help="Max allowed computed_binding_E (eV)")
    p.add_argument("--n_strucs", type=int, default=1000, help="Number of structures to randomly sample")
    p.add_argument("--fmax", type=float, default=0.05, help="Force convergence criterion (eV/Å)")
    p.add_argument("--steps", type=int, default=200, help="Max optimizer steps")
    p.add_argument("--cpu", action="store_true", help="Run on CPU")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling")
    return p.parse_args()


def load_target_energies(gen_strucs: str) -> dict[tuple[int, int], float]:
    csv_path = os.path.join(gen_strucs, "binding_energies.csv")
    targets = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["batch_idx"]), int(row["graph_idx"]))
            targets[key] = float(row["target_binding_E"])
    return targets


def collect_vasp_files(gen_strucs: str) -> list[tuple[int, int, str]]:
    """Return list of (batch_idx, graph_idx, filepath) for all .vasp files."""
    entries = []
    for path in glob.glob(os.path.join(gen_strucs, "*.vasp")):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        try:
            b, g = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        entries.append((b, g, path))
    return entries


def main():
    args = parse_args()

    targets = load_target_energies(args.gen_strucs)
    log.info(f"Loaded {len(targets)} target energies from {args.gen_strucs}/binding_energies.csv")

    all_entries = collect_vasp_files(args.gen_strucs)
    log.info(f"Found {len(all_entries)} .vasp files in {args.gen_strucs}")

    rng = random.Random(args.seed)
    sample = rng.sample(all_entries, min(args.n_strucs, len(all_entries)))
    sample.sort(key=lambda x: (x[0], x[1]))
    log.info(f"Sampled {len(sample)} structures (seed={args.seed})")

    log.info(f"Loading MLFF: {args.mlff_checkpoint}")
    with Silent():
        calculator = OCPCalculator(
            model_name=args.mlff_checkpoint,
            local_cache="./mlff_checkpoint_cache",
            cpu=args.cpu,
        )

    os.makedirs(args.rlxd_strucs, exist_ok=True)
    be_csv_path = os.path.join(args.rlxd_strucs, "binding_energies.csv")
    ec_csv_path = os.path.join(args.rlxd_strucs, "element_counts.csv")
    fieldnames = ["batch_idx", "graph_idx", "target_binding_E", "computed_binding_E", "binding_E_diff"]

    # element -> number of structures it appears in (at least once)
    element_structure_counts: dict[str, int] = {}

    with open(be_csv_path, "w", newline="") as be_f:
        writer = csv.DictWriter(be_f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx, graph_idx, path in tqdm(sample, desc="Relaxing"):
            target_E = targets.get((batch_idx, graph_idx))

            def _empty_row(reason: str = "") -> dict:
                if reason:
                    log.debug(f"  [{batch_idx}_{graph_idx}] skipped: {reason}")
                return {
                    "batch_idx": batch_idx,
                    "graph_idx": graph_idx,
                    "target_binding_E": _fmt(target_E) if target_E is not None else "",
                    "computed_binding_E": "",
                    "binding_E_diff": "",
                }

            # File may not exist if the structure was invalid during generation
            if not os.path.isfile(path):
                writer.writerow(_empty_row("file missing"))
                continue

            try:
                atoms = ase.io.read(path)
            except Exception as e:
                log.warning(f"Failed to read {path}: {e}")
                writer.writerow(_empty_row("read error"))
                continue

            # Track element composition across all loaded structures
            present_elements = set(atoms.get_chemical_symbols())
            for el in present_elements:
                element_structure_counts[el] = element_structure_counts.get(el, 0) + 1

            try:
                atoms_rlxd = relax_w_mlff(
                    atoms=atoms,
                    calculator=calculator,
                    fmax=args.fmax,
                    steps=args.steps,
                )
            except Exception as e:
                log.warning(f"Relaxation error for {batch_idx}_{graph_idx}: {e}")
                writer.writerow(_empty_row("relaxation error"))
                continue

            if atoms_rlxd is None:
                writer.writerow(_empty_row("did not converge"))
                continue

            try:
                computed_E = atoms_rlxd.get_potential_energy()
            except Exception as e:
                log.warning(f"Energy extraction failed for {batch_idx}_{graph_idx}: {e}")
                writer.writerow(_empty_row("energy extraction error"))
                continue

            if computed_E > args.energy_cutoff:
                writer.writerow(_empty_row(f"energy {computed_E:.4f} > cutoff {args.energy_cutoff}"))
                continue

            binding_E_diff = abs(computed_E - target_E) if target_E is not None else None
            writer.writerow({
                "batch_idx": batch_idx,
                "graph_idx": graph_idx,
                "target_binding_E": _fmt(target_E) if target_E is not None else "",
                "computed_binding_E": _fmt(computed_E),
                "binding_E_diff": _fmt(binding_E_diff) if binding_E_diff is not None else "",
            })

            out_path = os.path.join(args.rlxd_strucs, f"{batch_idx}_{graph_idx}.vasp")
            ase.io.write(out_path, atoms_rlxd)

    log.info(f"Binding energies written to {be_csv_path}")

    # Write element counts sorted by count descending
    sorted_elements = sorted(element_structure_counts.items(), key=lambda x: -x[1])
    with open(ec_csv_path, "w", newline="") as ec_f:
        writer = csv.DictWriter(ec_f, fieldnames=["element", "count"])
        writer.writeheader()
        for el, count in sorted_elements:
            writer.writerow({"element": el, "count": count})

    log.info(f"Element counts written to {ec_csv_path} ({len(sorted_elements)} distinct elements)")


if __name__ == "__main__":
    main()
