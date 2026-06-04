"""Evaluate forge-generated structure sets for uniqueness and binding energy.

Usage:
    python scripts/evaluate.py --gen_strucs ./gen_strucs
    python scripts/evaluate.py --gen_strucs ./gen_strucs --relax --n_strucs 500
"""

import csv
import glob
import logging
import os
import statistics
import time
from multiprocessing import Pool

import ase.io
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target energy loading
# ---------------------------------------------------------------------------

def load_target_energies(gen_strucs: str) -> dict[tuple[int, int], float]:
    csv_path = os.path.join(gen_strucs, "binding_energies.csv")
    if not os.path.isfile(csv_path):
        return {}
    targets = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["batch_idx"]), int(row["graph_idx"]))
            targets[key] = float(row["target_binding_E"])
    return targets


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_structure_files(directory: str) -> list[str]:
    """Return sorted structure files from directory (.vasp preferred, .traj fallback)."""
    vasp = sorted(glob.glob(os.path.join(directory, "*.vasp")))
    if vasp:
        log.info(f"Found {len(vasp)} .vasp files in {directory}")
        return vasp
    traj = sorted(glob.glob(os.path.join(directory, "*.traj")))
    if traj:
        log.info(f"Found {len(traj)} .traj files in {directory}")
        return traj
    raise FileNotFoundError(f"No .vasp or .traj files found in {directory}")


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------

def _load_full_atoms(path: str) -> tuple[str, "Atoms | None"]:
    """Worker: load full atoms (with adsorbate) and return (path, atoms)."""
    try:
        return path, ase.io.read(path)
    except Exception:
        return path, None


def load_atoms_parallel(files: list[str], num_workers: int) -> dict[str, "Atoms | None"]:
    """Load all structure files in parallel. Returns path → Atoms mapping."""
    log.info(f"Loading {len(files)} structures ({num_workers} workers)...")
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        pairs = list(tqdm(pool.imap(_load_full_atoms, files),
                          total=len(files), desc="Loading structures"))
    log.info(f"Loaded in {time.time() - t0:.1f}s")
    return dict(pairs)


def compute_uniqueness(atoms_map: dict[str, "Atoms | None"]) -> dict:
    """Structural uniqueness via StructureMatcher on slab-only structures.

    Accepts pre-loaded full Atoms objects (with adsorbate); strips tags==2 here.
    StructureMatcher runs in the main process.
    """
    t0 = time.time()
    structures = []
    for atoms in atoms_map.values():
        if atoms is None:
            continue
        try:
            slab = atoms[atoms.get_tags() != 2]
            if len(slab) == 0:
                continue
            structures.append(AseAtomsAdaptor().get_structure(slab))
        except Exception:
            continue
    load_time = time.time() - t0
    log.info(f"Converted {len(structures)}/{len(atoms_map)} slabs ({load_time:.1f}s)")

    log.info("Running StructureMatcher...")
    t1 = time.time()
    groups = StructureMatcher().group_structures(structures)
    match_time = time.time() - t1

    return {
        "total": len(structures),
        "unique": len(groups),
        "uniqueness": len(groups) / len(structures) if structures else 0.0,
        "load_time": load_time,
        "match_time": match_time,
    }


# ---------------------------------------------------------------------------
# Summarise sanitisation + relaxation results
# ---------------------------------------------------------------------------

def summarize_results(rows: list[dict]) -> dict:
    validity_rows = [r for r in rows if "is_valid" in r]

    validity = {}
    if validity_rows:
        n = len(validity_rows)
        validity = {
            "n": n,
            "is_valid":          sum(bool(r["is_valid"]) for r in validity_rows) / n,
            "is_valid_naive":    sum(bool(r["is_valid_naive"]) for r in validity_rows) / n,
            "vol_ok":            sum(bool(r["vol_ok"]) for r in validity_rows) / n,
            "cat_dist_ok":       sum(bool(r["cat_dist_ok"]) for r in validity_rows) / n,
            "cat_atom_types_ok": sum(bool(r["cat_atom_types_ok"]) for r in validity_rows) / n,
            "cat_ads_dist_ok":   sum(bool(r["cat_ads_dist_ok"]) for r in validity_rows) / n,
            "ads_dist_ok":       sum(bool(r["ads_dist_ok"]) for r in validity_rows) / n,
        }

    diffs = [float(r["binding_E_diff"]) for r in rows if r.get("binding_E_diff")]
    energy_stats = {}
    if diffs:
        energy_stats = {
            "n_relaxed": len(diffs),
            "n_total":   len(rows),
            "mean":      statistics.mean(diffs),
            "median":    statistics.median(diffs),
            "std":       statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "min":       min(diffs),
            "max":       max(diffs),
        }

    return {"validity": validity, "energy_stats": energy_stats}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(
    directory: str,
    uniqueness: dict | None,
    sanitisation: dict | None,
    n_strucs: int | None,
    seed: int | None,
) -> str:
    from datetime import datetime
    lines = [
        "=" * 50,
        "GACE Evaluation Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Directory : {directory}",
        "=" * 50,
    ]
    if uniqueness is not None:
        lines += [
            "",
            "Uniqueness",
            f"  Total loaded  : {uniqueness['total']}",
            f"  Unique groups : {uniqueness['unique']}",
            f"  Score         : {uniqueness['uniqueness']:.4f}",
            f"  Load time     : {uniqueness['load_time']:.1f}s",
            f"  Match time    : {uniqueness['match_time']:.1f}s",
        ]

    if sanitisation is not None:
        validity = sanitisation["validity"]
        energy_stats = sanitisation["energy_stats"]

        lines += ["", f"Sanitisation  (n={n_strucs}, seed={seed})"]
        if validity:
            lines += [
                f"  Structures checked : {validity['n']}",
                f"  Valid (strict)     : {validity['is_valid']:.4f}",
                f"  Valid (naive)      : {validity['is_valid_naive']:.4f}",
                f"  vol_ok             : {validity['vol_ok']:.4f}",
                f"  cat_dist_ok        : {validity['cat_dist_ok']:.4f}",
                f"  cat_atom_types_ok  : {validity['cat_atom_types_ok']:.4f}",
                f"  cat_ads_dist_ok    : {validity['cat_ads_dist_ok']:.4f}",
                f"  ads_dist_ok        : {validity['ads_dist_ok']:.4f}",
            ]

        if energy_stats:
            lines += [
                "",
                f"  Binding Energy  ({energy_stats['n_relaxed']} / {energy_stats['n_total']} relaxed)",
                f"  Mean ΔE   : {energy_stats['mean']:.4f} eV",
                f"  Median ΔE : {energy_stats['median']:.4f} eV",
                f"  Std ΔE    : {energy_stats['std']:.4f} eV",
                f"  Min ΔE    : {energy_stats['min']:.4f} eV",
                f"  Max ΔE    : {energy_stats['max']:.4f} eV",
            ]

    lines += ["", "=" * 50]
    return "\n".join(lines)


def write_report(report: str, output_path: str) -> None:
    print(report)
    with open(output_path, "w") as f:
        f.write(report + "\n")
    log.info(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate forge-generated structures.")
    p.add_argument("--strucs_for_eval", help="Directory of generated structure files")
    p.add_argument("--check_novelty", default=False, action="store_true", help="Run structural uniqueness check via StructureMatcher")
    p.add_argument("--sanitize", default=False, action="store_true",
                   help="Run validity checks (and optionally MLFF relaxation) on structures")
    p.add_argument("--relax", default=False, action="store_true", help="Run MLFF relaxation step")
    p.add_argument("--n_strucs", type=int, default=None, help="Number of structures to process (default: all)")
    p.add_argument("--mlff_checkpoint", default="EquiformerV2-31M-S2EF-OC20-All+MD")
    p.add_argument("--energy_cutoff", type=float, default=10.0, help="Max binding energy (eV)")
    p.add_argument("--fmax", type=float, default=0.05, help="Force convergence criterion (eV/Å)")
    p.add_argument("--steps", type=int, default=200, help="Max optimizer steps")
    p.add_argument("--catalyst_dist_cutoff", type=float, default=0.75, help="Catalyst inter-atom distance cutoff (fraction of covalent radii sum)")
    p.add_argument("--cpu", action="store_true", help="Force CPU for MLFF")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8, help="Parallel workers for structure loading")
    p.add_argument("--relaxed_strucs", default="./relaxed_strucs", help="Output directory for relaxed structures")
    p.add_argument("--sanitized_strucs", default="./sanitized_strucs", help="Output directory for sanitized structures")
    p.add_argument("--output", default="./report.txt", help="Path for output report")
    return p.parse_args()


def main(**kwargs):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    files = collect_structure_files(args.strucs_for_eval)
    preloaded = load_atoms_parallel(files, args.num_workers)

    uniqueness = compute_uniqueness(preloaded) if args.check_novelty else None

    sanitisation = None
    if args.sanitize or args.relax:
        from gace.utils.catalyst_forge import CatalystForge

        calculator = None
        if args.relax:
            from fairchem.core import OCPCalculator
            from gace.utils.relaxation import Silent
            log.info(f"Loading MLFF: {args.mlff_checkpoint}")
            with Silent():
                calculator = OCPCalculator(
                    model_name=args.mlff_checkpoint,
                    local_cache="./mlff_checkpoint_cache",
                    cpu=args.cpu,
                )

        binding_energies = load_target_energies(gen_strucs=args.strucs_for_eval)
        if binding_energies:
            log.info(f"Loaded {len(binding_energies)} target energies")

        cf = CatalystForge(
            binding_energies=binding_energies,
            batch_idx=0,
            calculator=calculator,
            catalyst_dist_cutoff=args.catalyst_dist_cutoff,
            sanitize_otf=args.sanitize,
            relax_otf=args.relax,
            energy_cutoff=args.energy_cutoff,
            fmax=args.fmax,
            steps=args.steps,
            generated_strucs=args.strucs_for_eval,
            relaxed_strucs=args.relaxed_strucs,
            sanitized_strucs=args.sanitized_strucs,
            preloaded_atoms=preloaded,
            n_strucs=args.n_strucs,
            seed=args.seed,
        )
        rows = cf.run()
        sanitisation = summarize_results(rows)

    report = format_report(
        directory=args.strucs_for_eval,
        uniqueness=uniqueness,
        sanitisation=sanitisation,
        n_strucs=args.n_strucs if (args.sanitize or args.relax) else None,
        seed=args.seed if (args.sanitize or args.relax) else None,
    )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
