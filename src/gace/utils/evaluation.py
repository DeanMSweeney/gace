"""Evaluate forge-generated structure sets for uniqueness and binding energy.

Usage:
    python scripts/evaluate.py --gen_strucs ./gen_strucs
    python scripts/evaluate.py --gen_strucs ./gen_strucs --relax --n_strucs 500
"""

import csv
import glob
import logging
import os
import random
import statistics
import time
from multiprocessing import Pool

import ase.io
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from gace.utils.relaxation import _fmt, relax_w_mlff, Silent

log = logging.getLogger(__name__)

NON_METALS = {"H", "O", "N", "C"}


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

def _load_slab_structure(path):
    """Worker: strip adsorbate (tags==2) and return a pymatgen Structure."""
    try:
        atoms = ase.io.read(path)
        slab = atoms[atoms.get_tags() != 2]
        if len(slab) == 0:
            return None
        return AseAtomsAdaptor().get_structure(slab)
    except Exception:
        return None


def compute_uniqueness(files: list[str], num_workers: int) -> dict:
    """Structural uniqueness via StructureMatcher on slab-only structures.

    File loading is parallelised; StructureMatcher runs in the main process.
    """
    log.info(f"Uniqueness: loading {len(files)} structures ({num_workers} workers)...")
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_load_slab_structure, files),
                            total=len(files), desc="Loading slabs"))
    structures = [s for s in results if s is not None]
    load_time = time.time() - t0
    log.info(f"Loaded {len(structures)}/{len(files)} ({load_time:.1f}s)")

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
# Relaxation
# ---------------------------------------------------------------------------

def _parse_file_key(path: str) -> tuple[int, int] | None:
    """Extract (batch_idx, graph_idx) from a filename like '3_12.vasp'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _load_target_energies(gen_strucs: str) -> dict[tuple[int, int], float]:
    csv_path = os.path.join(gen_strucs, "binding_energies.csv")
    if not os.path.isfile(csv_path):
        return {}
    targets = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["batch_idx"]), int(row["graph_idx"]))
            targets[key] = float(row["target_binding_E"])
    return targets


def relax_structures(
    files: list[str],
    gen_strucs: str,
    rlxd_strucs: str,
    calculator,
    fmax: float,
    steps: int,
    energy_cutoff: float,
    n_strucs: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """Relax a random sample of structures and write results to rlxd_strucs/.

    Returns (rows, element_counts).
    """
    targets = _load_target_energies(gen_strucs)
    if targets:
        log.info(f"Loaded {len(targets)} target energies")

    n = len(files) if n_strucs is None else min(n_strucs, len(files))
    rng = random.Random(seed)
    sample = rng.sample(files, n)
    sample.sort()
    log.info(f"Relaxing {len(sample)} structures (seed={seed})")

    os.makedirs(rlxd_strucs, exist_ok=True)
    rows: list[dict] = []
    element_counts: dict[str, int] = {}

    for path in tqdm(sample, desc="Relaxing"):
        key = _parse_file_key(path)
        batch_idx, graph_idx = key if key else (None, None)
        target_E = targets.get(key) if key else None

        base_row = {
            "batch_idx": batch_idx,
            "graph_idx": graph_idx,
            "target_binding_E": _fmt(target_E) if target_E is not None else "",
            "computed_binding_E": "",
            "binding_E_diff": "",
        }

        try:
            atoms = ase.io.read(path)
        except Exception as e:
            log.warning(f"Failed to read {path}: {e}")
            rows.append(base_row)
            continue

        for el in set(atoms.get_chemical_symbols()):
            element_counts[el] = element_counts.get(el, 0) + 1

        try:
            atoms_rlxd = relax_w_mlff(atoms=atoms, calculator=calculator,
                                       fmax=fmax, steps=steps)
        except Exception as e:
            log.warning(f"Relaxation error for {path}: {e}")
            rows.append(base_row)
            continue

        if atoms_rlxd is None:
            rows.append(base_row)
            continue

        try:
            computed_E = atoms_rlxd.get_potential_energy()
        except Exception as e:
            log.warning(f"Energy extraction failed for {path}: {e}")
            rows.append(base_row)
            continue

        if computed_E > energy_cutoff:
            rows.append(base_row)
            continue

        diff = abs(computed_E - target_E) if target_E is not None else None
        rows.append({
            **base_row,
            "computed_binding_E": _fmt(computed_E),
            "binding_E_diff": _fmt(diff) if diff is not None else "",
        })
        stem = os.path.splitext(os.path.basename(path))[0]
        ase.io.write(os.path.join(rlxd_strucs, f"{stem}.vasp"), atoms_rlxd)

    be_path = os.path.join(rlxd_strucs, "binding_energies.csv")
    with open(be_path, "w", newline="") as f:
        fieldnames = ["batch_idx", "graph_idx", "target_binding_E", "computed_binding_E", "binding_E_diff"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ec_path = os.path.join(rlxd_strucs, "element_counts.csv")
    with open(ec_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["element", "count"])
        writer.writeheader()
        for el, count in sorted(element_counts.items(), key=lambda x: -x[1]):
            writer.writerow({"element": el, "count": count})

    log.info(f"Wrote CSVs to {rlxd_strucs}")
    return rows, element_counts


def summarize_relaxation(rows: list[dict], element_counts: dict[str, int]) -> dict:
    diffs = [float(r["binding_E_diff"]) for r in rows if r["binding_E_diff"]]
    metals = sorted(
        [(el, c) for el, c in element_counts.items() if el not in NON_METALS],
        key=lambda x: -x[1],
    )
    stats = {}
    if diffs:
        stats = {
            "n_relaxed": len(diffs),
            "n_total": len(rows),
            "mean": statistics.mean(diffs),
            "median": statistics.median(diffs),
            "std": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "min": min(diffs),
            "max": max(diffs),
        }
    return {"stats": stats, "top_metals": metals[:5]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(
    directory: str,
    uniqueness: dict,
    relaxation: dict | None,
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
        "",
        "Uniqueness",
        f"  Total loaded  : {uniqueness['total']}",
        f"  Unique groups : {uniqueness['unique']}",
        f"  Score         : {uniqueness['uniqueness']:.4f}",
        f"  Load time     : {uniqueness['load_time']:.1f}s",
        f"  Match time    : {uniqueness['match_time']:.1f}s",
    ]

    if relaxation is not None:
        stats = relaxation["stats"]
        metals = relaxation["top_metals"]
        lines += ["", f"Relaxation  (n={n_strucs}, seed={seed})"]
        if stats:
            lines += [
                f"  Relaxed   : {stats['n_relaxed']} / {stats['n_total']}",
                f"  Mean ΔE   : {stats['mean']:.4f} eV",
                f"  Median ΔE : {stats['median']:.4f} eV",
                f"  Std ΔE    : {stats['std']:.4f} eV",
                f"  Min ΔE    : {stats['min']:.4f} eV",
                f"  Max ΔE    : {stats['max']:.4f} eV",
            ]
        if metals:
            lines += ["", "  Top-5 metals"]
            for i, (el, count) in enumerate(metals, 1):
                lines.append(f"    {i}. {el:<4s}  {count} structures")

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
    p.add_argument("--gen_strucs", default="./gen_strucs", help="Directory of generated structure files")
    p.add_argument("--relax", default=False, action="store_true", help="Run MLFF relaxation step")
    p.add_argument("--rlxd_strucs", default="./rlxd_strucs", help="Output directory for relaxed structures")
    p.add_argument("--n_strucs", type=int, default=None, help="Number of structures to relax (default: all)")
    p.add_argument("--mlff_checkpoint", default="EquiformerV2-31M-S2EF-OC20-All+MD")
    p.add_argument("--energy_cutoff", type=float, default=10.0, help="Max binding energy (eV)")
    p.add_argument("--fmax", type=float, default=0.05, help="Force convergence criterion (eV/Å)")
    p.add_argument("--steps", type=int, default=200, help="Max optimizer steps")
    p.add_argument("--cpu", action="store_true", help="Force CPU for MLFF")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8, help="Parallel workers for uniqueness")
    p.add_argument("--output", default="./eval_report.txt", help="Path for output report")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    files = collect_structure_files(args.gen_strucs)

    uniqueness = compute_uniqueness(files, args.num_workers)

    relaxation = None
    if args.relax:
        from fairchem.core import OCPCalculator
        log.info(f"Loading MLFF: {args.mlff_checkpoint}")
        with Silent():
            calculator = OCPCalculator(
                model_name=args.mlff_checkpoint,
                local_cache="./mlff_checkpoint_cache",
                cpu=args.cpu,
            )
        rows, element_counts = relax_structures(
            files=files,
            gen_strucs=args.gen_strucs,
            rlxd_strucs=args.rlxd_strucs,
            calculator=calculator,
            fmax=args.fmax,
            steps=args.steps,
            energy_cutoff=args.energy_cutoff,
            n_strucs=args.n_strucs,
            seed=args.seed,
        )
        relaxation = summarize_relaxation(rows, element_counts)

    report = format_report(
        directory=args.gen_strucs,
        uniqueness=uniqueness,
        relaxation=relaxation,
        n_strucs=args.n_strucs if args.relax else None,
        seed=args.seed if args.relax else None,
    )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
