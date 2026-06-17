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
import pathlib 

import ase.io
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_structure_files(directory: str) -> list[str]:
    """Return sorted structure files from directory (.vasp preferred, .traj fallback)."""
    traj = sorted(str(p) for p in pathlib.Path(directory).rglob("*.traj"))
    log.info(f"Found {len(traj)} .traj files in {directory}")
    
    if not traj:
        raise FileNotFoundError(f"No .traj files found in directory: {directory}")
    
    return traj

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


def _atoms_to_slab_structures(atoms_iter) -> list:
    """Convert an iterable of full Atoms to slab-only pymatgen Structures.

    Strips adsorbate atoms (tags == 2); skips None / empty / unconvertible slabs.
    """
    structures = []
    for a in atoms_iter:
        if a is None:
            continue
        try:
            slab = a[a.get_tags() != 2]
            if len(slab) == 0:
                continue
            structures.append(AseAtomsAdaptor().get_structure(slab))
        except Exception:
            continue
    return structures


def compute_uniqueness(atoms: list) -> dict:
    """Batch structural uniqueness over a list of Atoms.

    Strips adsorbate (tags == 2), groups slabs via StructureMatcher, and returns
    a uniqueness ratio dict e.g. {'uniqueness (100)': 0.85}.
    """
    structures = _atoms_to_slab_structures(atoms)
    groups = StructureMatcher().group_structures(structures)
    uniqueness = len(groups) / len(structures) if structures else 0.0
    return {f"uniqueness ({len(structures)})": uniqueness}


def compute_novelty(atoms: list, reference_atoms: list) -> dict:
    """Fraction of generated slabs absent from a reference set (e.g. OC20).

    A generated slab is novel if it matches no reference slab under
    StructureMatcher. Each slab is compared independently against the fixed
    reference, so this measures pure novelty (no inter-generated dedup).
    Returns e.g. {'novelty (100)': 0.42}.
    """
    matcher = StructureMatcher()
    gen_structures = _atoms_to_slab_structures(atoms)
    ref_structures = _atoms_to_slab_structures(reference_atoms)
    log.info(f"Novelty: {len(gen_structures)} generated vs {len(ref_structures)} reference slabs")

    n_novel = 0
    for struc in tqdm(gen_structures, desc="Novelty check"):
        if not any(matcher.fit(ref, struc) for ref in ref_structures):
            n_novel += 1
    novelty = n_novel / len(gen_structures) if gen_structures else 0.0
    return {f"novelty ({len(gen_structures)})": novelty}


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
        threshold_multipliers = [1, 2, 3, 5, 10]
        within_pct = {}
        for m in threshold_multipliers:
            key = f"within_{m}T"
            flagged = [r[key] for r in rows if r.get(key) is not None]
            within_pct[key] = sum(flagged) / len(flagged) if flagged else 0.0
        energy_stats = {
            "n_relaxed": len(diffs),
            "n_total":   len(rows),
            "mean":      statistics.mean(diffs),
            "median":    statistics.median(diffs),
            "std":       statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "min":       min(diffs),
            "max":       max(diffs),
            **within_pct,
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
    novelty: dict | None = None,
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
        lines += ["", "Uniqueness"]
        for key, val in uniqueness.items():
            lines.append(f"  {key} : {val:.4f}")

    if novelty is not None:
        lines += ["", "Novelty"]
        for key, val in novelty.items():
            lines.append(f"  {key} : {val:.4f}")

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
                "",
                "  Within threshold (T=0.10 eV)",
                f"  ≤ 1×T (0.10 eV) : {energy_stats['within_1T']:.2%}",
                f"  ≤ 2×T (0.20 eV) : {energy_stats['within_2T']:.2%}",
                f"  ≤ 3×T (0.30 eV) : {energy_stats['within_3T']:.2%}",
                f"  ≤ 5×T (0.50 eV) : {energy_stats['within_5T']:.2%}",
                f"  ≤10×T (1.00 eV) : {energy_stats['within_10T']:.2%}",
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
    p.add_argument("--check_diversity", default=False, action="store_true", help="Run structural uniqueness check via StructureMatcher")
    p.add_argument("--check_novelty_against", default=None, help="Directory of reference structures (e.g. OC20); reports fraction of generated slabs absent from it")
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
    gen_atoms = list(preloaded.values())

    # Uniqueness and novelty are pure post-processing passes over the loaded set.
    uniqueness = compute_uniqueness(gen_atoms) if args.check_diversity else None

    novelty = None
    if args.check_novelty_against:
        ref_files = collect_structure_files(args.check_novelty_against)
        ref_atoms = list(load_atoms_parallel(ref_files, args.num_workers).values())
        novelty = compute_novelty(gen_atoms, ref_atoms)

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

        cf = CatalystForge(
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
        rows = cf.fire()
        sanitisation = summarize_results(rows)

    report = format_report(
        directory=args.strucs_for_eval,
        novelty=novelty,
        uniqueness=uniqueness,
        sanitisation=sanitisation,
        n_strucs=args.n_strucs if (args.sanitize or args.relax) else None,
        seed=args.seed if (args.sanitize or args.relax) else None,
    )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
