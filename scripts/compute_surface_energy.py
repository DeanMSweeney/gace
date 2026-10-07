#!/usr/bin/env python
"""
Surface energy for the primitive cells in .traj files.

    python compute_surface_energies.py --traj_dirs run1/ run2/ --out surface_energy.csv

For each structure the primitive cell is taken from atoms.info (written by
gace.utils.builder.build_slab_adsorbate) and relaxed with the cell held fixed to
give the bulk reference. The bare surface is then built from that *relaxed*
primitive cell and its supercell matrix M — same builder, empty adsorbate — and
relaxed with the subsurface atoms (tag 0) fixed, so the frozen interior sits at
the same geometry the bulk reference was computed at.

    gamma = (E_slab - n * E_prim) / (2 * A)

n = |det(M)| is the number of primitive cells in the supercell and A = |a x b|
is the area of the exposed face. The factor 2 is because adding vacuum along c
exposes both the top face and the bottom one across the periodic boundary.

The primitive cell carries no vacuum — the vacuum is added after the supercell
expansion, since M would otherwise scale it — so E_prim is a bulk energy. Both
energies come from the same checkpoint, so its total-energy convention cancels.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from tqdm import tqdm

from gace.utils.builder import build_slab_adsorbate
from gace.utils.relaxation import relax_w_mlff


def load_primitive_cell(atoms):
    """The primitive cell stored in atoms.info by build_slab_adsorbate.

    A .traj round-trip may hand it back as an Atoms or as the plain dict ASE's
    json encoder made from Atoms.todict(); both are accepted.
    """
    prim = atoms.info["primitive_cell"]
    if isinstance(prim, Atoms):
        return prim.copy()
    if isinstance(prim, dict):
        return Atoms(numbers=np.asarray(prim["numbers"]),
                     positions=np.asarray(prim["positions"]),
                     cell=np.asarray(prim["cell"]),
                     pbc=prim.get("pbc", True))
    raise TypeError(f"primitive_cell is a {type(prim).__name__}, expected Atoms or dict")


def load_transformation_matrix(atoms):
    """The (3, 3) integer supercell matrix stored in atoms.info."""
    M = np.round(np.asarray(atoms.info["transformation_matrix"], dtype=float)).astype(int)
    if M.shape != (3, 3):
        raise ValueError(f"transformation_matrix has shape {M.shape}, expected (3, 3)")
    return M


def build_bare_slab(prim, M, vacuum):
    """Expand the primitive cell by M and add vacuum, with no adsorbate.

    build_slab_adsorbate does the whole job — supercell, vacuum, surface/bulk
    tagging — as long as the adsorbate arrays are empty.
    """
    a, b, c, alpha, beta, gamma = prim.cell.cellpar()
    return build_slab_adsorbate(
        prim_pos=prim.get_positions(),
        prim_atomic_numbers=prim.get_atomic_numbers(),
        lattice_params=np.array([a, b, c, np.radians(alpha), np.radians(beta), np.radians(gamma)]),
        transformation_matrix=M,
        ads_pos=np.empty((0, 3)),
        ads_atomic_numbers=np.empty((0,), dtype=int),
        vacuum=vacuum,
    )


def fix_subsurface(atoms):
    """Constrain the atoms tagged 0 (below the surface layer)."""
    fixed = [a.index for a in atoms if a.tag == 0]
    if not fixed:
        raise ValueError("slab has no tag-0 atoms; every atom is within 2 A of the top")
    atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def surface_area(atoms):
    """Area of the exposed face, |a x b| in A^2.

    The vacuum step only scales cell[2], so a and b are exactly the supercell's
    in-plane vectors and the area is independent of the vacuum thickness.
    """
    cell = np.asarray(atoms.get_cell())
    return float(np.linalg.norm(np.cross(cell[0], cell[1])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("surface_energy.csv"))
    p.add_argument("--model_name", default="EquiformerV2-153M-OMAT24-MP-sAlex")
    p.add_argument("--vacuum", type=float, default=15.0)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # load structures
    structures, no_prim = {}, []
    for f in sorted(f for d in args.traj_dirs for f in d.glob("*.traj")):
        atoms = read(f, index=-1)
        if "primitive_cell" not in atoms.info:
            no_prim.append(f.stem)
            continue
        structures[f.stem] = (load_primitive_cell(atoms), load_transformation_matrix(atoms))
    print(f"{len(structures)} structures")
    if no_prim:
        print(f"skipped {len(no_prim)} without a primitive_cell: {no_prim[:10]}")

    from fairchem.core import OCPCalculator
    # local_cache is required by fairchem 1.10 when model_name is set, but for
    # Hugging Face models the checkpoint goes to the HF cache (HF_HOME) instead
    calc = OCPCalculator(model_name=args.model_name, local_cache="./checkpoints",
                         cpu=False, seed=args.seed)

    # per structure
    rows, unconverged = [], []
    for name, (prim, M) in tqdm(structures.items(), desc="gamma"):
        try:
            # |det(M)| and the atom-count ratio are independent routes to n; they
            # disagree only if the build went wrong, which would otherwise give a
            # plausible-looking but wrong gamma.
            n = int(round(abs(np.linalg.det(M))))

            # bulk reference: positions only, cell held fixed
            prim_relaxed, converged = relax_w_mlff(atoms=prim, calculator=calc, fmax=args.fmax,
                                                   steps=args.steps, optimizer="BFGS")
            if not converged:
                unconverged.append(name)
                continue
            e_bulk = prim_relaxed.get_potential_energy()

            # Bare surface, built from the RELAXED primitive cell: the fixed tag-0
            # interior then sits at exactly the geometry e_bulk was computed at.
            # Building from the unrelaxed cell would lock strain into the interior
            # that the reference does not carry, inflating gamma by n*dE/(2A).
            # The prim relaxation holds the cell fixed, so M, n and A are unchanged.
            slab = build_bare_slab(prim_relaxed, M, args.vacuum)
            if len(slab) != n * len(prim_relaxed):
                raise ValueError(f"slab has {len(slab)} atoms, expected {n} x {len(prim_relaxed)}")

            slab_relaxed, converged = relax_w_mlff(atoms=fix_subsurface(slab), calculator=calc,
                                                   fmax=args.fmax, steps=args.steps, optimizer="BFGS")
            if not converged:
                unconverged.append(name)
                continue
            e_slab = slab_relaxed.get_potential_energy()
            area = surface_area(slab_relaxed)
        except Exception as exc:
            tqdm.write(f"  {name}: {exc}")
            continue
        rows.append({
            "name": name,
            "bulk_energy": e_bulk,
            "slab_energy": e_slab,
            "area": area,
            "n": n,
            "surface_energy": (e_slab - n * e_bulk) / (2 * area),
        })

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "bulk_energy", "slab_energy",
                                           "area", "n", "surface_energy"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)}/{len(structures)} structures to {args.out}")
    if unconverged:
        print(f"excluded {len(unconverged)} unconverged (fmax={args.fmax}, "
              f"steps={args.steps}): {unconverged[:10]}")


if __name__ == "__main__":
    main()
