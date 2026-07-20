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
            slab = a #[a.get_tags() != 2]
            slab.center(vacuum=20, axis=2)
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


# Per-worker globals, populated once per process by _init_novelty_worker so the
# (large) reference set is shipped to each worker once rather than per task.
_NOVELTY_REF_STRUCTURES: list | None = None
_NOVELTY_MATCHER: "StructureMatcher | None" = None


def _init_novelty_worker(ref_structures: list) -> None:
    global _NOVELTY_REF_STRUCTURES, _NOVELTY_MATCHER
    _NOVELTY_REF_STRUCTURES = ref_structures
    _NOVELTY_MATCHER = StructureMatcher()


def _is_novel(struc) -> bool:
    """True if struc matches no reference slab."""
    return not any(_NOVELTY_MATCHER.fit(ref, struc) for ref in _NOVELTY_REF_STRUCTURES)


def compute_novelty(atoms: list, reference_atoms: list, num_workers: int = 8) -> dict:
    """Fraction of generated slabs absent from a reference set (e.g. OC20).

    A generated slab is novel if it matches no reference slab under
    StructureMatcher. Each slab is compared independently against the fixed
    reference, so this measures pure novelty (no inter-generated dedup).
    Generated slabs are checked in parallel across num_workers processes.
    Returns e.g. {'novelty (100)': 0.42}.
    """
    gen_structures = _atoms_to_slab_structures(atoms)
    ref_structures = _atoms_to_slab_structures(reference_atoms)
    log.info(f"Novelty: {len(gen_structures)} generated vs {len(ref_structures)} reference slabs "
             f"({num_workers} workers)")

    if not gen_structures:
        return {"novelty (0)": 0.0}

    workers = max(1, min(num_workers, len(gen_structures)))
    with Pool(workers, initializer=_init_novelty_worker, initargs=(ref_structures,)) as pool:
        flags = list(tqdm(pool.imap(_is_novel, gen_structures, chunksize=8),
                          total=len(gen_structures), desc="Novelty check"))
    n_novel = sum(flags)
    return {f"novelty ({len(gen_structures)})": n_novel / len(gen_structures)}


def compute_composition_novelty(atoms: list, reference_atoms: list) -> dict:
    """Fraction of generated slabs whose chemical formula is absent from the reference.

    Purely composition-based: ignores geometry, lattice, and vacuum entirely, so it
    is unaffected by the slab/vacuum limitations of StructureMatcher. Adsorbate atoms
    (tags == 2) are stripped first, consistent with compute_novelty, so this compares
    catalyst slab compositions only. Formulas are compared via reduced formula, which
    is order- and stoichiometry-invariant (e.g. Pd2Cu == CuPd2, Cu2Pd4 == CuPd2).

    Returns e.g. {'composition_novelty (500)': 0.12, 'unique_gen_formulas': 37,
                  'ref_formulas': 210}.
    """
    gen_structures = _atoms_to_slab_structures(atoms)
    ref_structures = _atoms_to_slab_structures(reference_atoms)
    if not gen_structures:
        return {"composition_novelty (0)": 0.0}

    ref_formulas = {s.composition.reduced_formula for s in ref_structures}
    gen_formulas = [s.composition.reduced_formula for s in gen_structures]
    n_novel = sum(f not in ref_formulas for f in gen_formulas)
    log.info(f"Composition novelty: {len(set(gen_formulas))} unique generated formulas, "
             f"{len(ref_formulas)} reference formulas, {n_novel} absent from reference")
    return {
        f"composition_novelty ({len(gen_formulas)})": n_novel / len(gen_formulas),
        "unique_gen_formulas": len(set(gen_formulas)),
        "ref_formulas": len(ref_formulas),
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
    composition_novelty: dict | None = None,
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

    if composition_novelty is not None:
        lines += ["", "Composition novelty"]
        for key, val in composition_novelty.items():
            lines.append(f"  {key} : {val:.4f}" if isinstance(val, float) else f"  {key} : {val}")

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
    p.add_argument("--strucs", help="Directory of generated structure files")
    p.add_argument("--dedupe", default=False, action="store_true", help="Run structural uniqueness check via StructureMatcher")
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

    files = collect_structure_files(args.strucs)
    preloaded = load_atoms_parallel(files, args.num_workers)
    gen_atoms = list(preloaded.values())

    # Uniqueness and novelty are pure post-processing passes over the loaded set.
    uniqueness = compute_uniqueness(gen_atoms) if args.dedupe else None

    novelty = None
    composition_novelty = None
    if args.check_novelty_against:
        ref_files = collect_structure_files(args.check_novelty_against)
        ref_atoms = list(load_atoms_parallel(ref_files, args.num_workers).values())
        novelty = compute_novelty(gen_atoms, ref_atoms, num_workers=args.num_workers)
        composition_novelty = compute_composition_novelty(gen_atoms, ref_atoms)

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
            generated_strucs=args.strucs,
            relaxed_strucs=args.relaxed_strucs,
            sanitized_strucs=args.sanitized_strucs,
            preloaded_atoms=preloaded,
            n_strucs=args.n_strucs,
            seed=args.seed,
        )
        rows = cf.fire()
        sanitisation = summarize_results(rows)

    report = format_report(
        directory=args.strucs,
        novelty=novelty,
        composition_novelty=composition_novelty,
        uniqueness=uniqueness,
        sanitisation=sanitisation,
        n_strucs=args.n_strucs if (args.sanitize or args.relax) else None,
        seed=args.seed if (args.sanitize or args.relax) else None,
    )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
