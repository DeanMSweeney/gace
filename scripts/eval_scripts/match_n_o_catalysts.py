#!/usr/bin/env python3
"""Find catalyst slabs present in both an N-binding and an O-binding structure set.

Strips the adsorbate (tags == 2) from every structure in each set and compares the
remaining slabs with pymatgen's StructureMatcher — the same rule the novelty check in
eval_process_strucs.py uses. Writes one CSV row per matching (N, O) pair; an N file
matching several O files produces several rows.

Usage:
    python scripts/eval_scripts/match_n_o_catalysts.py \
        --n_dir ./n_relaxed/1_threshold \
        --o_dir ./o_relaxed/1_threshold \
        --out ./n_o_matches.csv
"""

import argparse
import csv
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import ase.io
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

ADSORBATE_TAG = 2
VACUUM = 20.0  # Å; must match between the two sets or the fits are meaningless
CSV_DECIMALS = 6
MAX_REPORTED_FAILURES = 5


@dataclass
class Record:
    """One structure, reduced to what the match pass and the CSV need."""
    name: str          # path relative to the set root
    structure: object  # pymatgen Structure, already primitive-reduced
    energy: float
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


# Set once per worker process by _init_load_worker.
_LOAD_ROOT: Path | None = None
_LOAD_ENERGY_KEY: str | None = None


def _init_load_worker(root: Path, energy_key: str) -> None:
    global _LOAD_ROOT, _LOAD_ENERGY_KEY
    _LOAD_ROOT = root
    _LOAD_ENERGY_KEY = energy_key


def _load_record(path: Path) -> "Record | str":
    """Worker: build a Record, or return a string describing why the file is unusable.

    The slab is reduced to its primitive cell here so the match pass compares
    already-reduced cells (see _matches_for_n). The vacuum padding keeps the facet
    and termination intact through the reduction.
    """
    name = str(path.relative_to(_LOAD_ROOT))
    try:
        atoms = ase.io.read(path)
    except Exception as e:
        return f"{name}: unreadable ({e})"

    slab = atoms[atoms.get_tags() != ADSORBATE_TAG]
    if len(slab) == 0:
        return f"{name}: no slab atoms (every atom tagged as adsorbate)"
    if _LOAD_ENERGY_KEY not in atoms.info:
        return f"{name}: missing atoms.info['{_LOAD_ENERGY_KEY}']"

    slab.center(vacuum=VACUUM, axis=2)
    structure = AseAtomsAdaptor().get_structure(slab)
    matcher = StructureMatcher()
    reduced = matcher._get_reduced_structure(structure, matcher._primitive_cell, niggli=True)
    return Record(
        name=name,
        structure=reduced,
        energy=float(atoms.info[_LOAD_ENERGY_KEY]),
        formula=reduced.composition.reduced_formula,
    )


def load_records(root: Path, energy_key: str, num_workers: int, label: str) -> list[Record]:
    """Load every .traj under root. Raises if any file is unusable."""
    files = collect(root)
    with Pool(
        processes=num_workers,
        initializer=_init_load_worker,
        initargs=(root, energy_key),
    ) as pool:
        results = list(tqdm(pool.imap(_load_record, files),
                            total=len(files), desc=f"Loading {label} set"))

    failures = [r for r in results if isinstance(r, str)]
    if failures:
        shown = "\n  ".join(failures[:MAX_REPORTED_FAILURES])
        more = f"\n  ... and {len(failures) - MAX_REPORTED_FAILURES} more" \
            if len(failures) > MAX_REPORTED_FAILURES else ""
        raise RuntimeError(
            f"{len(failures)}/{len(files)} files under {root} are unusable:\n  {shown}{more}"
        )
    return results


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Set once per worker process by _init_match_worker so the O set is shipped to each
# worker once rather than once per task.
_O_STRUCTURES: list | None = None
_MATCHER: "StructureMatcher | None" = None


def _init_match_worker(o_structures: list) -> None:
    global _O_STRUCTURES, _MATCHER
    _O_STRUCTURES = o_structures
    _MATCHER = StructureMatcher()


def _matches_for_n(n_structure) -> list[int]:
    """Indices of every O slab matching this N slab.

    Both sides were reduced at load time, so fit() skips structure reduction — the
    dominant cost of a comparison.
    """
    return [
        i for i, o_structure in enumerate(_O_STRUCTURES)
        if _MATCHER.fit(o_structure, n_structure, skip_structure_reduction=True)
    ]


def find_matches(n_records: list[Record], o_records: list[Record],
                 num_workers: int) -> list[tuple[Record, Record]]:
    """All (N, O) pairs whose slabs match under StructureMatcher."""
    o_structures = [r.structure for r in o_records]
    n_structures = [r.structure for r in n_records]

    workers = max(1, min(num_workers, len(n_structures)))
    with Pool(workers, initializer=_init_match_worker, initargs=(o_structures,)) as pool:
        hits = list(tqdm(pool.imap(_matches_for_n, n_structures, chunksize=8),
                         total=len(n_structures), desc="Matching N vs O"))

    return [(n_records[n_i], o_records[o_i])
            for n_i, o_indices in enumerate(hits)
            for o_i in o_indices]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(pairs: list[tuple[Record, Record]], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_filename", "o_filename",
            "n_computed_binding_E", "o_computed_binding_E",
            "reduced_formula",
        ])
        for n_rec, o_rec in pairs:
            # A fit under StructureMatcher defaults (primitive_cell, scale) implies equal
            # reduced formulas; a mismatch means that assumption broke.
            if n_rec.formula != o_rec.formula:
                raise RuntimeError(
                    f"Matched slabs disagree on reduced formula: "
                    f"{n_rec.name} is {n_rec.formula}, {o_rec.name} is {o_rec.formula}"
                )
            writer.writerow([
                n_rec.name,
                o_rec.name,
                format(n_rec.energy, f".{CSV_DECIMALS}f"),
                format(o_rec.energy, f".{CSV_DECIMALS}f"),
                n_rec.formula,
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_dir", type=Path, required=True,
                        help="Root directory of N-binding .traj files (searched recursively)")
    parser.add_argument("--o_dir", type=Path, required=True,
                        help="Root directory of O-binding .traj files (searched recursively)")
    parser.add_argument("--out", type=Path, default=Path("n_o_matches.csv"),
                        help="Output CSV path")
    parser.add_argument("--energy_key", default="computed_binding_E",
                        help="atoms.info key holding the binding energy")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    n_records = load_records(root=args.n_dir, energy_key=args.energy_key,
                             num_workers=args.num_workers, label="N")
    o_records = load_records(root=args.o_dir, energy_key=args.energy_key,
                             num_workers=args.num_workers, label="O")

    pairs = find_matches(n_records=n_records, o_records=o_records,
                         num_workers=args.num_workers)
    write_csv(pairs=pairs, out_path=args.out)

    print(f"N set: {len(n_records)} structures from {args.n_dir}")
    print(f"O set: {len(o_records)} structures from {args.o_dir}")
    print(f"Wrote {args.out}: {len(pairs)} matching pairs "
          f"({len({n.name for n, _ in pairs})} distinct N files, "
          f"{len({o.name for _, o in pairs})} distinct O files)")


if __name__ == "__main__":
    main()
