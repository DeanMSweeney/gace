#!/usr/bin/env python3
"""Find catalyst slabs present in two structure sets with different adsorbates.

Strips the adsorbate (tags == 2) from every structure in each set and compares the
remaining slabs with pymatgen's StructureMatcher. With --invar_surface_matcher, each slab
is first rotated so its surface normal is +z and compared with SurfaceMatcher (from
gace.utils.invar_surface_matcher, shared with eval_process_strucs.py). Writes one CSV row per matching (A, B) pair; an A file
matching several B files produces several rows. Every distinct matched A file is also
copied into --unique_A_dir. With --rebuild_from_prim, both sets' slabs are instead
rebuilt from atoms.info's primitive cell and supercell matrix M before comparison.

Usage (writes n_o_matches_metadata.csv and matching_catalyst_by_n/ by default):
    python scripts/match_catalysts.py \
        --A_dir ./n_relaxed/1_threshold ./n_relaxed/2_threshold --A_label N \
        --B_dir ./o_relaxed/1_threshold ./o_relaxed/2_threshold --B_label O
"""

import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from gace.utils.builder import build_slab_adsorbate
from gace.utils.invar_surface_matcher import SurfaceMatcher, canonical_repr

ADSORBATE_TAG = 2
VACUUM = 20.0  # Å; must match between the two sets or the fits are meaningless
CSV_DECIMALS = 6
MAX_REPORTED_FAILURES = 5


@dataclass
class Record:
    """One structure, reduced to what the match pass and the CSV need."""
    name: str          # path relative to the set root
    path: Path         # full path to the .traj file
    structure: object  # pymatgen Structure, already primitive-reduced
    energy: float | None  # None when atoms.info lacks the energy key
    formula: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def collect(root: Path) -> list[Path]:
    """Return sorted .traj files under root."""
    files = sorted(root.rglob("*.traj"))
    if not files:
        raise FileNotFoundError(f"No .traj files found under {root}")
    return files


def _rebuild_slab_from_prim(atoms: Atoms) -> Atoms:
    """Rebuild the bare slab from atoms.info's primitive cell and supercell matrix M.

    Same as eval_process_strucs.py. A .traj round-trip may return the primitive cell
    as Atoms or as an Atoms.todict() dict.
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
        vacuum=VACUUM,
    )


def _load_atoms(root_and_path: tuple[Path, Path]) -> "tuple[str, Path, Atoms | str]":
    """Worker: read one .traj; returns (name, path, atoms), or an error string in place of atoms."""
    root, path = root_and_path
    name = str(path.relative_to(root))
    try:
        return name, path, ase.io.read(path)
    except Exception as e:
        return name, path, f"{name}: unreadable ({e})"


# Set once per worker process by _init_build_worker.
_LOAD_ENERGY_KEY: str | None = None
_REBUILD_FROM_PRIM: bool = False
_INVAR_SURFACE_MATCHER: bool = False


def _init_build_worker(energy_key: str, rebuild_from_prim: bool,
                       invar_surface_matcher: bool) -> None:
    global _LOAD_ENERGY_KEY, _REBUILD_FROM_PRIM, _INVAR_SURFACE_MATCHER
    _LOAD_ENERGY_KEY = energy_key
    _REBUILD_FROM_PRIM = rebuild_from_prim
    _INVAR_SURFACE_MATCHER = invar_surface_matcher


def _build_slab(job: "tuple[str, Path, Atoms]") -> "tuple[str, Path, Atoms, float | None] | str":
    """Worker: bare, vacuum-padded slab (rebuilt from prim + M if set) and its energy,
    or a string describing why the structure is unusable."""
    name, path, atoms = job
    if _REBUILD_FROM_PRIM:
        missing = [k for k in ("primitive_cell", "transformation_matrix") if k not in atoms.info]
        if missing:
            return f"{name}: --rebuild_from_prim needs atoms.info keys {missing}"
        slab = _rebuild_slab_from_prim(atoms)
    else:
        slab = atoms[atoms.get_tags() != ADSORBATE_TAG]
    if len(slab) == 0:
        return f"{name}: no slab atoms (every atom tagged as adsorbate)"
    energy = atoms.info.get(_LOAD_ENERGY_KEY)
    slab.center(vacuum=VACUUM, axis=2)
    return name, path, slab, None if energy is None else float(energy)


def _build_record(job: "tuple[str, Path, Atoms, float | None]") -> "Record | str":
    """Worker: canonicalize (if set) and reduce a built slab into a Record.

    The slab is reduced to its primitive cell here so the match pass compares
    already-reduced cells (see _matches_for_a). The vacuum padding keeps the facet
    and termination intact through the reduction.
    """
    name, path, slab, energy = job
    if _INVAR_SURFACE_MATCHER:
        try:
            structure = canonical_repr(slab)
        except ValueError as e:
            return f"{name}: {e}"
    else:
        structure = AseAtomsAdaptor().get_structure(slab)
    reduced = StructureMatcher._get_reduced_structure(structure, primitive_cell=True, niggli=True)
    return Record(
        name=name,
        path=path,
        structure=reduced,
        energy=energy,
        formula=reduced.composition.reduced_formula,
    )


def _raise_failures(failures: list[str], total: int, label: str) -> None:
    if not failures:
        return
    shown = "\n  ".join(failures[:MAX_REPORTED_FAILURES])
    more = f"\n  ... and {len(failures) - MAX_REPORTED_FAILURES} more" \
        if len(failures) > MAX_REPORTED_FAILURES else ""
    raise RuntimeError(f"{len(failures)}/{total} {label} files are unusable:\n  {shown}{more}")


def load_records(roots: list[Path], energy_key: str, rebuild_from_prim: bool,
                 invar_surface_matcher: bool, num_workers: int, label: str) -> list[Record]:
    """Load every .traj under each root, then build its slab. Raises if any file is unusable."""
    files = [(root, path) for root in roots for path in collect(root)]
    with Pool(processes=num_workers) as pool:
        loaded = list(tqdm(pool.imap(_load_atoms, files),
                           total=len(files), desc=f"Loading {label}"))
    _raise_failures(failures=[a for _, _, a in loaded if isinstance(a, str)],
                    total=len(files), label=label)

    with Pool(
        processes=num_workers,
        initializer=_init_build_worker,
        initargs=(energy_key, rebuild_from_prim, invar_surface_matcher),
    ) as pool:
        # Bars shown only for the optional stages: Building (rebuild) and Canonicalizing.
        slabs = list(tqdm(pool.imap(_build_slab, loaded, chunksize=8), total=len(loaded),
                          desc=f"Building {label}", disable=not rebuild_from_prim))
        _raise_failures(failures=[s for s in slabs if isinstance(s, str)],
                        total=len(files), label=label)
        results = list(tqdm(pool.imap(_build_record, slabs, chunksize=8), total=len(slabs),
                            desc=f"Canonicalizing {label}", disable=not invar_surface_matcher))
    _raise_failures(failures=[r for r in results if isinstance(r, str)],
                    total=len(files), label=label)
    return results


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Set once per worker process by _init_match_worker so the B set is shipped to each
# worker once rather than once per task.
_B_STRUCTURES: list | None = None
_MATCHER: "StructureMatcher | None" = None


def _init_match_worker(b_structures: list, invar_surface_matcher: bool) -> None:
    global _B_STRUCTURES, _MATCHER
    _B_STRUCTURES = b_structures
    _MATCHER = SurfaceMatcher() if invar_surface_matcher else StructureMatcher()


def _matches_for_a(a_structure) -> list[int]:
    """Indices of every B slab matching this A slab.

    Both sides were reduced at load time, so fit() skips structure reduction — the
    dominant cost of a comparison.
    """
    return [
        i for i, b_structure in enumerate(_B_STRUCTURES)
        if _MATCHER.fit(b_structure, a_structure, skip_structure_reduction=True)
    ]


def find_matches(a_records: list[Record], b_records: list[Record], invar_surface_matcher: bool,
                 num_workers: int, a_label: str, b_label: str) -> list[tuple[Record, Record]]:
    """All (A, B) pairs whose slabs match under StructureMatcher (SurfaceMatcher with invar_surface_matcher)."""
    b_structures = [r.structure for r in b_records]
    a_structures = [r.structure for r in a_records]

    workers = max(1, min(num_workers, len(a_structures)))
    with Pool(workers, initializer=_init_match_worker,
              initargs=(b_structures, invar_surface_matcher)) as pool:
        hits = list(tqdm(pool.imap(_matches_for_a, a_structures, chunksize=8),
                         total=len(a_structures), desc=f"Matching {a_label} vs {b_label}"))

    return [(a_records[a_i], b_records[b_i])
            for a_i, b_indices in enumerate(hits)
            for b_i in b_indices]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _format_energy(energy: float | None) -> str:
    """Empty cell when the structure had no energy."""
    return "" if energy is None else format(energy, f".{CSV_DECIMALS}f")


def write_csv(pairs: list[tuple[Record, Record]], out_path: Path,
              a_label: str, b_label: str) -> None:
    a, b = a_label.lower(), b_label.lower()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{a}_filename", f"{b}_filename",
            f"{a}_computed_binding_E", f"{b}_computed_binding_E",
            "reduced_formula",
        ])
        for a_rec, b_rec in pairs:
            # A fit under StructureMatcher defaults (primitive_cell, scale) implies equal
            # reduced formulas; a mismatch means that assumption broke.
            if a_rec.formula != b_rec.formula:
                raise RuntimeError(
                    f"Matched slabs disagree on reduced formula: "
                    f"{a_rec.name} is {a_rec.formula}, {b_rec.name} is {b_rec.formula}"
                )
            writer.writerow([
                a_rec.name,
                b_rec.name,
                _format_energy(a_rec.energy),
                _format_energy(b_rec.energy),
                a_rec.formula,
            ])


def copy_unique_a(pairs: list[tuple[Record, Record]], out_dir: Path,
                  num_workers: int) -> int:
    """Copy each distinct matched A file into out_dir, keeping its relative name.

    Threads, not processes: the copy is I/O-bound.
    """
    unique = {a_rec.path: a_rec for a_rec, _ in pairs}
    names = [rec.name for rec in unique.values()]
    if len(set(names)) != len(names):
        raise RuntimeError("Distinct A files share a relative name across --A_dir roots; "
                           "copying them flat into one directory would overwrite files")
    jobs = [(rec.path, out_dir / rec.name) for rec in unique.values()]
    for parent in {dst.parent for _, dst in jobs}:
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # list() so any copy error is raised here
        list(tqdm(pool.map(lambda job: shutil.copy2(*job), jobs),
                  total=len(jobs), desc="Copying matched A files"))
    return len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--A_dir", type=Path, nargs="+", required=True,
                        help="Root directories of set-A .traj files (searched recursively)")
    parser.add_argument("--B_dir", type=Path, nargs="+", required=True,
                        help="Root directories of set-B .traj files (searched recursively)")
    parser.add_argument("--A_label", required=True,
                        help="Name of set A's adsorbate, e.g. N; used in CSV columns and logs")
    parser.add_argument("--B_label", required=True,
                        help="Name of set B's adsorbate, e.g. O; used in CSV columns and logs")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: <a>_<b>_matches_metadata.csv)")
    parser.add_argument("--unique_A_dir", type=Path, default=None,
                        help="Directory to copy every distinct matched A .traj into "
                             "(default: matching_catalyst_by_<a>)")
    parser.add_argument("--energy_key", default="computed_binding_E",
                        help="atoms.info key holding the binding energy")
    parser.add_argument("--rebuild_from_prim", default=False, action="store_true",
                        help="Rebuild both sets' slabs from atoms.info's primitive cell and M "
                             "instead of using stored positions")
    parser.add_argument("--invar_surface_matcher", default=False, action="store_true",
                        help="Canonicalize slabs (surface normal along +z) and match with "
                             "SurfaceMatcher instead of StructureMatcher")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()
    a, b = args.A_label.lower(), args.B_label.lower()
    if args.out is None:
        args.out = Path(f"{a}_{b}_matches_metadata.csv")
    if args.unique_A_dir is None:
        args.unique_A_dir = Path(f"matching_catalyst_by_{a}")

    a_records = load_records(roots=args.A_dir, energy_key=args.energy_key,
                             rebuild_from_prim=args.rebuild_from_prim,
                             invar_surface_matcher=args.invar_surface_matcher,
                             num_workers=args.num_workers, label=args.A_label)
    b_records = load_records(roots=args.B_dir, energy_key=args.energy_key,
                             rebuild_from_prim=args.rebuild_from_prim,
                             invar_surface_matcher=args.invar_surface_matcher,
                             num_workers=args.num_workers, label=args.B_label)

    pairs = find_matches(a_records=a_records, b_records=b_records,
                         invar_surface_matcher=args.invar_surface_matcher,
                         num_workers=args.num_workers,
                         a_label=args.A_label, b_label=args.B_label)
    write_csv(pairs=pairs, out_path=args.out, a_label=args.A_label, b_label=args.B_label)
    n_copied = copy_unique_a(pairs=pairs, out_dir=args.unique_A_dir,
                             num_workers=args.num_workers)

    print(f"{args.A_label} set: {len(a_records)} structures from "
          f"{', '.join(map(str, args.A_dir))}")
    print(f"{args.B_label} set: {len(b_records)} structures from "
          f"{', '.join(map(str, args.B_dir))}")
    print(f"Wrote {args.out}: {len(pairs)} matching pairs "
          f"({len({a.name for a, _ in pairs})} distinct {args.A_label} files, "
          f"{len({b.name for _, b in pairs})} distinct {args.B_label} files)")
    print(f"Copied {n_copied} distinct {args.A_label} files to {args.unique_A_dir}")


if __name__ == "__main__":
    main()
