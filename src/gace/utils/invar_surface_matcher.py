"""Rotation-invariant slab matching: canonicalize surface normal to +z, then match without flips."""

import numpy as np
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

SURFACE_TAG = 1


class SurfaceMatcher(StructureMatcher):
    """Rejects lattice mappings that reverse the surface normal (+z after canonicalization)."""
    def _get_lattices(self, target_lattice, s, supercell_size=1):
        for latt, sc_m in super()._get_lattices(target_lattice, s, supercell_size):
            R = np.linalg.solve(target_lattice.matrix, latt.matrix)
            if R[2, 2] > 0:
                yield latt, sc_m


def surface_normal(atoms: Atoms, surface_mask: np.ndarray) -> np.ndarray:
    """Unit normal to cell[0] x cell[1], pointing toward the surface atoms."""
    n = np.cross(atoms.cell[0], atoms.cell[1])
    n /= np.linalg.norm(n)
    h = atoms.positions @ n
    if h[surface_mask].mean() < h.mean():
        n = -n
    return n


def rotation_to_z(cell: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Rotation with rows (e1, e2, n): sends n to +z and cell[0] to +x."""
    e1 = cell[0] / np.linalg.norm(cell[0])
    e2 = np.cross(n, e1)
    return np.array([e1, e2, n])


def canonical_repr(slab: Atoms) -> Structure:
    """Rotate the slab so its surface normal (toward tag-1 atoms) points along +z; return a pymatgen Structure."""
    surface_mask = slab.get_tags() == SURFACE_TAG
    if not surface_mask.any():
        raise ValueError(f"Slab {slab.get_chemical_formula()} has no surface atoms (tag {SURFACE_TAG})")
    cell = np.asarray(slab.cell)
    R = rotation_to_z(cell=cell, n=surface_normal(slab, surface_mask=surface_mask))
    return Structure(Lattice(cell @ R.T), slab.get_chemical_symbols(),
                     slab.positions @ R.T, coords_are_cartesian=True)
