#!/usr/bin/env python3
"""Evaluate forge-generated structure sets for uniqueness and binding energy.

Usage:
    python scripts/eval_scripts/eval_process_strucs.py --traj_dirs ./gen_strucs
    python scripts/eval_scripts/eval_process_strucs.py --traj_dirs ./run_a ./run_b --relax --n_strucs 500
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
import numpy as np
from ase import Atoms
from tqdm import tqdm

from gace.utils.builder import build_slab_adsorbate
from gace.utils.invar_surface_matcher import SurfaceMatcher, canonical_repr
from pymatgen.analysis.structure_matcher import StructureMatcher

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_structure_files(directories: str | list[str]) -> list[str]:
    """Return sorted .traj files across one or more directories."""
    if isinstance(directories, str):
        directories = [directories]

    files: list[str] = []
    for d in directories:
        found = sorted(str(p) for p in pathlib.Path(d).rglob("*.traj"))
        if not found:
            raise FileNotFoundError(f"No .traj files found in directory: {d}")
        log.info(f"Found {len(found)} .traj files in {d}")
        files.extend(found)

    log.info(f"Found {len(files)} .traj files across {len(directories)} directories")
    return files

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


def _rebuild_slab_from_prim(atoms: Atoms) -> Atoms:
    """Rebuild the bare slab from atoms.info's primitive cell and supercell matrix M.

    A .traj round-trip may return the primitive cell as Atoms or as an Atoms.todict() dict.
    """
    prim = atoms.info["primitive_cell"]
    if isinstance(prim, dict):
        prim = Atoms(numbers=np.asarray(prim["numbers"]),
                     positions=np.asarray(prim["positions"]),
                     cell=np.asarray(prim["cell"]),
                     pbc=prim.get("pbc", True))
    elif not isinstance(prim, Atoms):
        raise TypeError(f"primitive_cell is a {type(prim).__name__}, expected Atoms or dict")

    M = np.asarray(atoms.info["transformation_matrix"], dtype=float)
    a, b, c, alpha, beta, gamma = prim.cell.cellpar()
    return build_slab_adsorbate(
        prim_pos=prim.get_positions(),
        prim_atomic_numbers=prim.get_atomic_numbers(),
        lattice_params=np.array([a, b, c, np.radians(alpha), np.radians(beta), np.radians(gamma)]),
        transformation_matrix=M,
        ads_pos=np.empty((0, 3)),
        ads_atomic_numbers=np.empty((0,), dtype=int),
        vacuum=20.0,
    )


def rebuild_slabs_parallel(atoms: list, num_workers: int) -> list:
    """Rebuild bare slabs from primitive cell + M in parallel; skips None entries."""
    atoms = [a for a in atoms if a is not None]
    log.info(f"Rebuilding {len(atoms)} slabs from primitive cell ({num_workers} workers)...")
    with Pool(processes=num_workers) as pool:
        return list(tqdm(pool.imap(_rebuild_slab_from_prim, atoms, chunksize=8),
                         total=len(atoms), desc="Rebuilding slabs"))


def _atoms_to_slab_structure(a: "Atoms | None"):
    """Convert full Atoms to a canonicalized slab-only Structure; None if None / empty / unconvertible."""
    if a is None:
        return None
    try:
        slab = a[a.get_tags() != 2]
        slab.center(vacuum=20, axis=2)
    except Exception:
        return None
    if len(slab) == 0:
        return None
    return canonical_repr(slab)


def _atoms_to_slab_structures(atoms_iter) -> list:
    """Convert an iterable of full Atoms to slab-only pymatgen Structures.

    Strips adsorbate atoms (tags == 2); skips None / empty / unconvertible slabs.
    """
    structures = (_atoms_to_slab_structure(a) for a in atoms_iter)
    return [s for s in structures if s is not None]


def compute_uniqueness(atoms: list) -> dict:
    """Batch structural uniqueness over a list of Atoms.

    Strips adsorbate (tags == 2), groups slabs via StructureMatcher, and returns
    a uniqueness ratio dict e.g. {'uniqueness (100)': 0.85}.
    """
    structures = _atoms_to_slab_structures(atoms)
    groups = StructureMatcher().group_structures(structures)
    uniqueness = len(groups) / len(structures) if structures else 0.0
    return {f"uniqueness ({len(structures)})": uniqueness}


def compute_composition_uniqueness(atoms: list) -> dict:
    """Fraction of distinct slab reduced formulas among generated slabs.

    Composition-only counterpart to compute_uniqueness; adsorbate (tags == 2) stripped.
    Returns e.g. {'composition_uniqueness (500)': 0.07}.
    """
    structures = _atoms_to_slab_structures(atoms)
    if not structures:
        return {"composition_uniqueness (0)": 0.0}
    formulas = {s.composition.reduced_formula for s in structures}
    return {f"composition_uniqueness ({len(structures)})": len(formulas) / len(structures)}


# Per-worker globals, populated once per process by _init_novelty_worker so the
# (large) reference set is shipped to each worker once rather than per task.
_NOVELTY_REF_STRUCTURES: list | None = None
_NOVELTY_MATCHER: "SurfaceMatcher | None" = None


def _init_novelty_worker(ref_structures: list) -> None:
    global _NOVELTY_REF_STRUCTURES, _NOVELTY_MATCHER
    _NOVELTY_REF_STRUCTURES = ref_structures
    _NOVELTY_MATCHER = SurfaceMatcher()


def _is_novel(struc) -> bool:
    """True if struc matches no reference slab."""
    return not any(_NOVELTY_MATCHER.fit(ref, struc) for ref in _NOVELTY_REF_STRUCTURES)


def compute_novelty(atoms: list, paths: list[str], reference_atoms: list,
                    num_workers: int = 8) -> tuple[dict, list[dict]]:
    """Fraction of generated slabs absent from a reference set (e.g. OC20).

    A generated slab is novel if it matches no reference slab under
    StructureMatcher. Each slab is compared independently against the fixed
    reference, so this measures pure novelty (no inter-generated dedup).
    Generated slabs are checked in parallel across num_workers processes.
    paths must align with atoms; unconvertible slabs are dropped from both outputs.
    Returns ({'novelty (100)': 0.42}, per-file rows of path / formula / novel / composition_novel).
    """
    if len(atoms) != len(paths):
        raise ValueError(f"atoms ({len(atoms)}) and paths ({len(paths)}) are misaligned")
    pairs = [(p, _atoms_to_slab_structure(a)) for p, a in zip(paths, atoms)]
    pairs = [(p, s) for p, s in pairs if s is not None]
    gen_structures = [s for _, s in pairs]
    ref_structures = _atoms_to_slab_structures(reference_atoms)
    log.info(f"Novelty: {len(gen_structures)} generated vs {len(ref_structures)} reference slabs "
             f"({num_workers} workers)")

    if not gen_structures:
        return {"novelty (0)": 0.0}, []

    workers = max(1, min(num_workers, len(gen_structures)))
    with Pool(workers, initializer=_init_novelty_worker, initargs=(ref_structures,)) as pool:
        flags = list(tqdm(pool.imap(_is_novel, gen_structures, chunksize=8),
                          total=len(gen_structures), desc="Novelty check"))

    ref_formulas = {s.composition.reduced_formula for s in ref_structures}
    rows = []
    for (path, s), novel in zip(pairs, flags):
        formula = s.composition.reduced_formula
        rows.append({"path": path, "formula": formula, "novel": novel,
                     "composition_novel": formula not in ref_formulas})

    n_novel = sum(flags)
    return {f"novelty ({len(gen_structures)})": n_novel / len(gen_structures)}, rows


def write_novelty_csv(rows: list[dict], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "formula", "novel", "composition_novel"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Per-file novelty written to {output_path}")


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

# Reported in this order; each is a per-stage flag written by CatalystForge.
# A value of None means the check did not run for that structure (its stage was
# disabled, or an earlier check already failed), so it is excluded from both the
# numerator and the denominator — rates are conditional on the check running.
_CHECK_KEYS = ["is_valid", "is_valid_base", "is_unique", "mlff_converged", "passed_all_checks"]


def summarize_results(rows: list[dict]) -> dict:
    checks = {}
    for key in _CHECK_KEYS:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if vals:
            checks[key] = {"n": len(vals), "rate": sum(bool(v) for v in vals) / len(vals)}

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

    return {"checks": checks, "energy_stats": energy_stats}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(
    directories: list[str],
    uniqueness: dict | None,
    sanitisation: dict | None,
    n_strucs: int | None,
    seed: int | None,
    novelty: dict | None = None,
    composition_novelty: dict | None = None,
    composition_uniqueness: dict | None = None,
) -> str:
    from datetime import datetime
    lines = [
        "=" * 50,
        "GACE Evaluation Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Directories : {', '.join(directories)}",
        "=" * 50,
    ]
    if uniqueness is not None:
        lines += ["", "Uniqueness"]
        for key, val in uniqueness.items():
            lines.append(f"  {key} : {val:.4f}")

    if composition_uniqueness is not None:
        lines += ["", "Composition uniqueness"]
        for key, val in composition_uniqueness.items():
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
        checks = sanitisation["checks"]
        energy_stats = sanitisation["energy_stats"]

        lines += ["", f"Structure checks  (n={n_strucs}, seed={seed})"]
        if checks:
            width = max(len(k) for k in checks)
            for key, stat in checks.items():
                lines.append(f"  {key:<{width}} : {stat['rate']:.4f}   (n={stat['n']})")

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
    p.add_argument("--traj_dirs", nargs="+", help="One or more directories of generated structure files")
    p.add_argument("--eval_uniqueness", default=False, action="store_true", help="Run structural uniqueness check")
    p.add_argument("--eval_novelty_against", nargs="+", default=None, help="One or more directories of reference structures (e.g. OC20); reports fraction of generated slabs absent from them")
    p.add_argument("--rebuild_from_prim", default=False, action="store_true", help="For uniqueness/novelty, rebuild generated slabs from atoms.info's primitive cell and M instead of using stored positions")
    p.add_argument("--sanitize", default=False, action="store_true", help="Run sanitisation checks")
    p.add_argument("--dedupe", default=False, action="store_true", help="Mark duplicate slabs (records is_unique); requires --sanitize")
    p.add_argument("--relax", default=False, action="store_true", help="Relax structures with the MLFF and report binding energies")
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
    p.add_argument("--output", default="./eval_report.txt", help="Path for output report")
    p.add_argument("--novelty_csv", default="./novelty.csv", help="Per-file novelty CSV (written with --eval_novelty_against)")
    return p.parse_args()


def main(**kwargs):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    files = collect_structure_files(args.traj_dirs)
    preloaded = load_atoms_parallel(files, args.num_workers)
    # Drop failed loads here so gen_paths stays aligned with gen_atoms through the rebuild.
    gen_paths = [p for p, a in preloaded.items() if a is not None]
    gen_atoms = [preloaded[p] for p in gen_paths]
    # Only the generated set is rebuilt; reference structures and sanitize/relax use stored positions.
    if args.rebuild_from_prim:
        gen_atoms = rebuild_slabs_parallel(gen_atoms, args.num_workers)

    # Uniqueness and novelty are pure post-processing passes over the loaded set.
    uniqueness = compute_uniqueness(gen_atoms) if args.eval_uniqueness else None
    composition_uniqueness = compute_composition_uniqueness(gen_atoms) if args.eval_uniqueness else None

    novelty = None
    composition_novelty = None
    if args.eval_novelty_against:
        ref_files = collect_structure_files(args.eval_novelty_against)
        ref_atoms = list(load_atoms_parallel(ref_files, args.num_workers).values())
        novelty, novelty_rows = compute_novelty(gen_atoms, paths=gen_paths, reference_atoms=ref_atoms,
                                                num_workers=args.num_workers)
        write_novelty_csv(rows=novelty_rows, output_path=args.novelty_csv)
        composition_novelty = compute_composition_novelty(gen_atoms, ref_atoms)

    sanitisation = None
    if args.sanitize or args.relax or args.dedupe:
        from gace.utils.forge import CatalystForge

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
            dedupe_otf=args.dedupe,
            relax_otf=args.relax,
            energy_cutoff=args.energy_cutoff,
            fmax=args.fmax,
            steps=args.steps,
            generated_strucs=None,  # only read on the opt_graph (OTF) path
            relaxed_strucs=args.relaxed_strucs,
            sanitized_strucs=args.sanitized_strucs,
            preloaded_atoms=preloaded,
            n_strucs=args.n_strucs,
            seed=args.seed,
        )
        rows = cf.fire()
        sanitisation = summarize_results(rows)

    report = format_report(
        directories=args.traj_dirs,
        novelty=novelty,
        composition_novelty=composition_novelty,
        uniqueness=uniqueness,
        composition_uniqueness=composition_uniqueness,
        sanitisation=sanitisation,
        n_strucs=args.n_strucs if (args.sanitize or args.relax or args.dedupe) else None,
        seed=args.seed if (args.sanitize or args.relax or args.dedupe) else None,
    )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
