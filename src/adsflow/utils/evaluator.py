"""Structural validity evaluation for forge-generated structures."""

import numpy as np
from ase import Atoms


def check_validity(atoms: Atoms) -> dict:
    """Check structural validity of a single ASE Atoms object.

    Runs two checks:
    - Volume: cell volume >= 0.1 Å³ (guards against degenerate cells)
    - Distance: minimum interatomic distance >= 0.5 Å (guards against atom clashes)

    Args:
        atoms: ASE Atoms object to check.

    Returns:
        dict with keys: vol_ok (bool), dist_ok (bool), is_valid (bool).
    """
    # Volume check
    try:
        vol_ok = float(atoms.get_volume()) >= 0.1
    except Exception:
        vol_ok = False

    # Atom clash check
    try:
        if len(atoms) > 1:
            dists = atoms.get_all_distances(mic=True)
            min_dist = np.min(dists[np.nonzero(dists)])
            dist_ok = bool(min_dist >= 0.5)
        else:
            dist_ok = True
    except Exception:
        dist_ok = False

    return {
        "vol_ok": vol_ok,
        "dist_ok": dist_ok,
        "is_valid": vol_ok and dist_ok,
    }


def evaluate_forge_batch(atoms_list: list) -> dict:
    """Evaluate structural validity across a batch of forge-generated structures.

    Args:
        atoms_list: List of ASE Atoms objects from build_forge_batch.

    Returns:
        dict with mean validity rates: vol_ok, dist_ok, is_valid (all floats in [0, 1]).
    """
    results = [check_validity(atoms) for atoms in atoms_list]
    return {
        "validity/vol_ok": float(np.mean([r["vol_ok"] for r in results])),
        "validity/dist_ok": float(np.mean([r["dist_ok"] for r in results])),
        "validity/is_valid": float(np.mean([r["is_valid"] for r in results])),
    }
