#!/usr/bin/env python
"""
Energy above hull for the primitive cells in .traj files.

Needs .ehull_cache/ from build_mp_cache.py. No network or mp_api required.

    python ehull_from_traj.py --traj_dir run1/ run2/ --out ehull.csv

Follows the Matbench Discovery convention: the raw MLIP energy, with MP2020
corrections, is compared against a fixed hull built from MP DFT energies.
Formation energies use MP's DFT elemental references.
"""
import argparse
import csv
import json
from pathlib import Path

from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm


def relax(atoms, calc, fmax, steps):
    atoms = atoms.copy()
    atoms.calc = calc
    converged = FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy(), atoms, bool(converged)


def load_cached(elements, cache_dir):
    """MP entries for a chemical system, rebuilt from plain JSON records."""
    path = cache_dir / f"{'-'.join(sorted(elements))}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not in cache; add it to chemsys.txt "
                                f"and rerun build_mp_cache.py")
    return [ComputedEntry(Composition(r["composition"]), r["energy"], entry_id=r["entry_id"])
            for r in json.loads(path.read_text())]


def mp_parameters(struct):
    """run_type / hubbards matching what MP would have used for this composition."""
    incar = MPRelaxSet(struct).incar
    if incar.get("LDAU", False):
        return {"run_type": "GGA+U",
                "hubbards": dict(zip([e.symbol for e in struct.composition.elements],
                                     incar.get("LDAUU", [])))}
    return {"run_type": "GGA", "hubbards": {}}


def build_entry(struct, energy, name):
    entry = ComputedStructureEntry(structure=struct, energy=energy,
                                   parameters=mp_parameters(struct), entry_id=name)
    MaterialsProject2020Compatibility(check_potcar=False).process_entries([entry])
    return entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("ehull.csv"))
    p.add_argument("--model_name", default="EquiformerV2-153M-OMAT24-MP-sAlex")
    p.add_argument("--cache_dir", type=Path, default=Path(".ehull_cache"))
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # load structures
    structures = {}
    for f in sorted(f for d in args.traj_dir for f in d.glob("*.traj")):
        prim = read(f, index=-1).info.get("primitive_cell")
        if prim is not None:
            structures[f.stem] = (prim if hasattr(prim, "get_positions")
                                  else AseAtomsAdaptor.get_atoms(prim))
    elements = sorted({s for a in structures.values() for s in a.get_chemical_symbols()})
    print(f"{len(structures)} structures, {len(elements)} elements")

    from fairchem.core import OCPCalculator
    # local_cache is required by fairchem 1.10 when model_name is set, but for
    # Hugging Face models the checkpoint goes to the HF cache (HF_HOME) instead
    calc = OCPCalculator(model_name=args.model_name, local_cache="./checkpoints", cpu=False, seed=args.seed)

    # per structure
    hulls, rows, unconverged = {}, [], []
    for name, atoms in tqdm(structures.items(), desc="E_hull"):
        try:
            key = tuple(sorted(set(atoms.get_chemical_symbols())))
            if key not in hulls:
                hulls[key] = PhaseDiagram(load_cached(key, args.cache_dir))
            energy, relaxed, converged = relax(atoms, calc, args.fmax, args.steps)
            if not converged:
                unconverged.append(name)
                continue
            struct = AseAtomsAdaptor.get_structure(relaxed)
            entry = build_entry(struct, energy, name)
            ef = hulls[key].get_form_energy_per_atom(entry)
            decomp, e_hull = hulls[key].get_decomp_and_e_above_hull(entry, allow_negative=True)
        except Exception as exc:
            tqdm.write(f"  {name}: {exc}")
            continue
        rows.append({
            "name": name,
            "formula": struct.composition.reduced_formula,
            "e_form_per_atom": ef,
            "e_above_hull": e_hull,
            "decomposition": "; ".join(f"{e.composition.reduced_formula}:{a:.3f}"
                                       for e, a in decomp.items()),
        })

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "formula", "e_form_per_atom",
                                           "e_above_hull", "decomposition"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)}/{len(structures)} structures to {args.out}")
    if unconverged:
        print(f"excluded {len(unconverged)} unconverged (fmax={args.fmax}, "
              f"steps={args.steps}): {unconverged[:10]}")


if __name__ == "__main__":
    main()
