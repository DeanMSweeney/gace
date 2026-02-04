import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict

from adsflow.utils.invariance import pbc_aware_delta


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
        norm=False
    )
    per_atom_mae = per_atom_error.abs().mean(dim=1, keepdim=True)
    per_atom_me = per_atom_error.mean(dim=1, keepdim=True)

    # Adsorbate-specific errors
    ads_atom_error = pbc_aware_delta(
        delta=pos_predicted[adsorbate_mask] - pos_target[adsorbate_mask],
        cell=cell,
        batch=batch[adsorbate_mask],
        norm=False,
    )
    ads_atom_mae = ads_atom_error.abs().mean(dim=1, keepdim=True)
    ads_atom_me = ads_atom_error.mean(dim=1, keepdim=True)

    # Calculate per-structure metrics
    batch_ads = batch[adsorbate_mask]
    num_structures = batch.max().item() + 1
    mae_per_structure = []
    ads_mae_per_structure = []

    for struct_idx in range(num_structures):
        struct_mask = batch == struct_idx
        ads_mask = batch_ads == struct_idx
        if struct_mask.any():
            mae_per_structure.append(per_atom_mae[struct_mask].mean().item())
            ads_mae_per_structure.append(ads_atom_mae[ads_mask].mean().item())

    mae_per_structure = torch.tensor(mae_per_structure)
    ads_mae_per_structure = torch.tensor(ads_mae_per_structure)

    # Calculate DwT and ADwT
    thresholds = torch.arange(threshold_min, threshold + threshold_step, threshold_step)
    dwt_values = []
    ads_dwt_values = []

    for thresh in thresholds:
        dwt_values.append((mae_per_structure < thresh.item()).sum().item() / num_structures * 100.0)
        ads_dwt_values.append((ads_mae_per_structure < thresh.item()).sum().item() / num_structures * 100.0)

    return {
        "generated_mae": per_atom_mae.mean().item(),
        "generated_me": per_atom_me.mean().item(),
        "generated_dwt": dwt_values[-1],
        "generated_adwt": sum(dwt_values) / len(dwt_values),
        "generated_ads_mae": ads_atom_mae.mean().item(),
        "generated_ads_me": ads_atom_me.mean().item(),
        "generated_ads_dwt": ads_dwt_values[-1],
        "generated_ads_adwt": sum(ads_dwt_values) / len(ads_dwt_values),
    }
