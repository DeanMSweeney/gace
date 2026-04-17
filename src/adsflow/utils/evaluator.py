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


def check_validity(atoms: Atoms) -> dict:
    """Check structural validity of a single ASE Atoms object.

    Runs two checks:
    - Volume: cell volume >= 0.1 Å³ (guards against degenerate cells)
    - Distance: no pair closer than 50% of their summed covalent radii (guards against atom clashes)

    Args:
        atoms: ASE Atoms object to check, or None if build failed.

    Returns:
        dict with keys: vol_ok (bool), dist_ok (bool), is_valid (bool).
    """
    if atoms is None:
        return {"vol_ok": False, "dist_ok": False, "is_valid": False}

    # Volume check
    try:
        vol_ok = float(atoms.get_volume()) >= 0.1
    except Exception:
        vol_ok = False

    # Atom clash check
    # Toggle method: 1 = ASE fixed cutoff  2 = ASE covalent radii
    DIST_METHOD = 2
    try:
        if len(atoms) > 1:
            if DIST_METHOD == 1:
                dists = atoms.get_all_distances(mic=True)
                np.fill_diagonal(dists, np.inf)
                dist_ok = bool(dists.min() >= 0.5)

            elif DIST_METHOD == 2:
                dists = atoms.get_all_distances(mic=True)
                numbers = atoms.get_atomic_numbers()
                radii = covalent_radii[numbers]               # (N,)
                sum_radii = radii[:, None] + radii[None, :]   # (N, N)
                np.fill_diagonal(dists, np.inf)
                dist_ok = bool(np.all(dists >= 1 * sum_radii))
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

def compute_eval_metrics(
    data: Data,
    threshold: float = 0.5,
    threshold_min: float = 0.01,
    threshold_step: float = 0.001,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predicted atomic structures.

    Computes:
    - MAE: Mean Absolute Error of atom positions
    - ME: Mean Error (signed) of atom positions
    - DwT: Distance within Threshold (% structures with MAE below threshold)
    - ADwT: Average DwT across thresholds (robust to outliers)

    All metrics are computed for both the full system and adsorbate atoms only.

    Args:
        data: Data object with pos_generated, pos_relaxed, cell, batch, fixed, tags
        threshold: Maximum threshold in Angstroms (default: 0.5)
        threshold_min: Minimum threshold in Angstroms (default: 0.01)
        threshold_step: Step size for thresholds in Angstroms (default: 0.001)

    Returns:
        Dictionary containing all evaluation metrics
    """
    batch = data.batch
    cell = data.cell
    pos_target = data.pos_relaxed
    pos_predicted = data.pos_generated
    adsorbate_mask = (data.tags == 2)

    # Calculate PBC-aware per-atom errors
    per_atom_error = pbc_aware_delta(
        delta=pos_predicted - pos_target,
        cell=cell,
        batch=batch,
    )
    per_atom_mae = per_atom_error.abs().mean(dim=1, keepdim=True)
    per_atom_me = per_atom_error.mean(dim=1, keepdim=True)

    # Adsorbate-specific errors
    ads_atom_error = pbc_aware_delta(
        delta=pos_predicted[adsorbate_mask] - pos_target[adsorbate_mask],
        cell=cell,
        batch=batch[adsorbate_mask],
    )
    ads_atom_mae = ads_atom_error.abs().mean(dim=1, keepdim=True)
    ads_atom_me = ads_atom_error.mean(dim=1, keepdim=True)

    # Calculate per-structure metrics
    batch_ads = batch[adsorbate_mask]
    num_structures = batch.max().item() + 1
    mae_per_structure = []
    me_per_structure = []
    ads_mae_per_structure = []
    ads_me_per_structure = []

    for struct_idx in range(num_structures):
        struct_mask = batch == struct_idx
        ads_mask = batch_ads == struct_idx
        if struct_mask.any():
            mae_per_structure.append(per_atom_mae[struct_mask].mean().item())
            me_per_structure.append(per_atom_me[struct_mask].mean().item())
            ads_mae_per_structure.append(ads_atom_mae[ads_mask].mean().item())
            ads_me_per_structure.append(ads_atom_me[ads_mask].mean().item())

    mae_per_structure = torch.tensor(mae_per_structure)
    me_per_structure = torch.tensor(me_per_structure)
    ads_mae_per_structure = torch.tensor(ads_mae_per_structure)
    ads_me_per_structure = torch.tensor(ads_me_per_structure)

    # Calculate DwT and ADwT
    thresholds = torch.arange(threshold_min, threshold + threshold_step, threshold_step)
    dwt_values = []
    ads_dwt_values = []

    for thresh in thresholds:
        dwt_values.append((mae_per_structure < thresh.item()).sum().item() / num_structures * 100.0)
        ads_dwt_values.append((ads_mae_per_structure < thresh.item()).sum().item() / num_structures * 100.0)

    return {
        "generated_mae": mae_per_structure.mean().item(),
        "generated_me": me_per_structure.mean().item(),
        "generated_dwt": dwt_values[-1],
        "generated_adwt": sum(dwt_values) / len(dwt_values),
        "generated_ads_mae": ads_mae_per_structure.mean().item(),
        "generated_ads_me": ads_me_per_structure.mean().item(),
        "generated_ads_dwt": ads_dwt_values[-1],
        "generated_ads_adwt": sum(ads_dwt_values) / len(ads_dwt_values),
    }
