import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint
from typing import Optional, Literal, Union, Tuple, List

from adsflow.utils.invariance import pbc_aware_delta

def ADwT(
    data: Data,
    pos_predicted: Tensor,
    threshold: float = 0.5,
    threshold_min: float = 0.01,
    threshold_step: float = 0.001,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate Distance within Threshold (DwT) and Average DwT (ADwT).

    DwT is the percentage of structures with an atom position MAE below the threshold.
    ADwT is the average DwT across thresholds from threshold_min to threshold in
    increments of threshold_step. MAE is computed for free catalyst and adsorbate
    atom positions with PBC. ADwT is robust to outliers and better indicates overall
    prediction quality.

    Args:
        pos_predicted: Predicted atom positions
        pos_target: Target (relaxed) atom positions
        cell: Unit cell parameters for PBC
        batch: Batch indices for atoms
        threshold: Maximum threshold in Angstroms (default: 0.5)
        free_atoms: Optional mask for fixed atoms (True = fixed, exclude from calculation)
        threshold_min: Minimum threshold in Angstroms (default: 0.01)
        threshold_step: Step size for thresholds in Angstroms (default: 0.001)

    Returns:
        Tuple of (overall MAE, DwT, ADwT, overall ads MAE, ads DwT, ads ADwT,
                  system mean error, adsorbate mean error)
    """
    # assign properties 
    batch = data.batch
    cell = data.cell
    pos_target = data.pos_relaxed
    free_atoms = (1 - data.fixed).bool()  # Mask for free (non-frozen) atoms
    adsorbate_mask = (data.tags == 2)
    catalyst_mask = (data.tags == 1)
    # Calculate PBC-aware per-atom distances once
    per_atom_error = pbc_aware_delta(
        delta=pos_predicted[catalyst_mask] - pos_target[catalyst_mask],
        cell=cell,
        batch=batch[catalyst_mask],
        norm=False
    )
    per_atom_mae = per_atom_error.abs().mean(dim=1, keepdim=True)  # (N, 1)
    per_atom_me = per_atom_error.mean(dim=1, keepdim=True)  # (N, 1)

    ads_atom_error = pbc_aware_delta(
        delta=pos_predicted[adsorbate_mask] - pos_target[adsorbate_mask],
        cell=cell,
        batch=batch[adsorbate_mask],
        norm=False,
    )
    ads_atom_mae = ads_atom_error.abs().mean(dim=1, keepdim=True)  # (N, 1)
    ads_atom_me = ads_atom_error.mean(dim=1, keepdim=True)  # (N, 1)

    # Include only free atoms, mask frozen atoms
    if free_atoms is not None:
        per_atom_mae = per_atom_mae
        per_atom_me = per_atom_me
        batch_filtered = batch
        batch_ads_filtered = batch[adsorbate_mask]
        ads_atom_mae = ads_atom_mae # all adsorabte atoms are free atoms by definition
        ads_atom_me = ads_atom_me # all adsorabte atoms are free atoms by definition
    else:
        batch_filtered = batch

    # Calculate MAE per structure (only once)
    num_structures = batch_filtered.max().item() + 1
    mae_per_structure = []
    ads_mae_per_structure = []
    me_per_structure = []
    ads_me_per_structure = []

    for struct_idx in range(num_structures):
        struct_mask = batch_filtered == struct_idx
        ads_mask = batch_ads_filtered == struct_idx
        if struct_mask.any():
            mae = per_atom_mae[struct_mask].mean().item()
            mae_per_structure.append(mae)

            me = per_atom_me[struct_mask].mean().item()
            me_per_structure.append(me)

            ads_mae = ads_atom_mae[ads_mask].mean().item()
            ads_mae_per_structure.append(ads_mae)

            ads_me = ads_atom_me[ads_mask].mean().item()
            ads_me_per_structure.append(ads_me)

    mae_per_structure = torch.tensor(mae_per_structure)
    ads_mae_per_structure = torch.tensor(ads_mae_per_structure)
    me_per_structure = torch.tensor(me_per_structure)
    ads_me_per_structure = torch.tensor(ads_me_per_structure)

    # Overall MAE across all atoms
    overall_mae = per_atom_mae.mean().item()
    overall_ads_mae = ads_atom_mae.mean().item()
    overall_me = per_atom_me.mean().item()
    overall_ads_me = ads_atom_me.mean().item()

    # Calculate DwT values across all thresholds from threshold_min to threshold
    thresholds = torch.arange(threshold_min, threshold + threshold_step, threshold_step)
    dwt_values = []
    ads_dwt_values = []

    for thresh in thresholds:
        structures_below = (mae_per_structure < thresh.item()).sum().item()
        dwt_val = (structures_below / num_structures) * 100.0
        dwt_values.append(dwt_val)

        ads_structures_below = (ads_mae_per_structure < thresh.item()).sum().item()
        ads_dwt_val = (ads_structures_below / num_structures) * 100.0
        ads_dwt_values.append(ads_dwt_val)

    # DwT at the specified threshold is the last value
    dwt = dwt_values[-1]
    ads_dwt = ads_dwt_values[-1]

    # ADwT is the average across all thresholds
    adwt = sum(dwt_values) / len(dwt_values)
    ads_adwt = sum(ads_dwt_values) / len(ads_dwt_values)

    return overall_mae, dwt, adwt, overall_ads_mae, ads_dwt, ads_adwt, overall_me, overall_ads_me