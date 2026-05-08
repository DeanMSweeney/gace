"""Structural validity evaluation for forge-generated structures."""

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict

from adsflow.utils.invariance import pbc_aware_delta


def check_validity(atoms: Atoms, covalent_radii_scale: float = 0.75, target_ads_dists: np.ndarray = None) -> dict:
    """Check structural validity of a single ASE Atoms object.

    Runs two checks:
    - Volume: cell volume >= 0.1 Å³ (guards against degenerate cells)
    - Distance: no pair closer than covalent_radii_scale * sum of covalent radii

    Args:
        atoms: ASE Atoms object to check, or None if build failed.
        covalent_radii_scale: Fraction of summed covalent radii used as minimum allowed distance.

    Returns:
        dict with keys: vol_ok (bool), cat_dist_ok (bool), ads_dist_ok (bool), is_valid (bool).
    """
    if atoms is None:
        return {"vol_ok": False, "cat_dist_ok": False, "ads_dist_ok": False, "ads_dist_mae": None, "ads_dist_me": None, "is_valid": False}

    # Volume check # intertatomic distance check
    try:
        vol_ok = float(atoms.get_volume()) >= 0.1
        tags = atoms.get_tags()
        cat_mask = tags != 2 # N
        ads_mask = tags == 2 # N 

        cat_atoms = atoms[cat_mask]
        ads_atoms = atoms[ads_mask]
        dists = atoms.get_all_distances(mic=True) # N, N 
        np.fill_diagonal(dists, np.inf) # N, N

        # first adsorbate atoms
        ads_idx = np.where(tags == 2)[0]
        ads_dists = dists[ads_mask, :] # N_ads, N
        ads_dist_ok = bool(ads_dists.min() >= 0.5) # scalar

        # then catalyst atoms 
        cat_idx = np.where(cat_mask)[0]
        cat_dists = dists[np.ix_(cat_idx, cat_idx)] # N_cat, N_cat 
        cat_atomic_numbers = cat_atoms.get_atomic_numbers()
        atomic_radii = covalent_radii[cat_atomic_numbers]               # (N_cat,)
        sum_atomic_radii = atomic_radii[:, None] + atomic_radii[None, :]   # (N_cat, N_cat)
        sum_atomic_radii[sum_atomic_radii == 0.0] = np.inf
        cat_dist_ok = bool(np.all(cat_dists >= covalent_radii_scale * sum_atomic_radii))

        # interatomic distance of adsorbate atoms
        num_ads_atoms = ads_mask.sum()
        generated_ads_interatomic_dist = dists[np.ix_(ads_idx, ads_idx)].copy()  # N_ads, N_ads
        np.fill_diagonal(generated_ads_interatomic_dist, 0.0)  # undo inf from line 43; self-dist = 0
        target_ads_interatomic_dist = np.asarray(target_ads_dists)[:num_ads_atoms, :num_ads_atoms]  # N_ads, N_ads
        ads_dist_mae = float(np.abs(target_ads_interatomic_dist - generated_ads_interatomic_dist).mean())

    except Exception:
        cat_dist_ok = False
        vol_ok = False
        ads_dist_ok = False
        ads_dist_mae = None

    return {
        "vol_ok": vol_ok,
        "cat_dist_ok": cat_dist_ok,
        "ads_dist_ok": ads_dist_ok,
        "ads_dist_mae": ads_dist_mae,
        "is_valid": vol_ok and cat_dist_ok and ads_dist_ok,
    }


