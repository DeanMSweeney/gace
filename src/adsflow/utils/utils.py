import os

import numpy as np 
from ase import Atoms
from ase.io import read, write

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple
from adsflow.utils.invariance import pbc_aware_delta

def adsorbate_neighborhood_mask(
    data: Data,
    cutoff: float,
) -> Tensor:
    """
    Compute a boolean mask selecting adsorbate atoms and nearby catalyst atoms.

    For each graph in the batch, marks:
    - All adsorbate atoms (tags == 2)
    - Non-frozen catalyst atoms within `cutoff` Angstroms of any adsorbate atom
      (using PBC-aware minimum image distances)

    Fully vectorized — no per-graph loop.

    Args:
        data: Batched PyG Data object with pos, cell, batch, tags, fixed attributes
        cutoff: Distance threshold in Angstroms

    Returns:
        mask: Boolean tensor (N,) — True for atoms in the adsorbate neighborhood
    """

    pos = data.pos.float()
    batch = data.batch
    cell = data.cell.float()
    adsorbate = (data.tags == 2)
    not_fixed = (data.fixed == 0)

    # Start with all adsorbate atoms
    mask = adsorbate.clone()

    ads_idx = adsorbate.nonzero(as_tuple=True)[0]
    cat_idx = (~adsorbate & not_fixed).nonzero(as_tuple=True)[0]

    if len(ads_idx) == 0 or len(cat_idx) == 0:
        return mask

    ads_pos = pos[ads_idx]  # (Na, 3)
    cat_pos = pos[cat_idx]  # (Nc, 3)
    Nc, Na = len(cat_idx), len(ads_idx)

    # Pairwise deltas: (Nc, Na, 3)
    delta = cat_pos.unsqueeze(1) - ads_pos.unsqueeze(0)

    # PBC: use each catalyst atom's graph index for cell lookup
    cat_batch = batch[cat_idx]
    pair_batch = cat_batch.unsqueeze(1).expand(-1, Na).reshape(-1)

    delta_pbc = pbc_aware_delta(
        cell=cell,
        delta=delta.reshape(-1, 3),
        batch=pair_batch,
        l2_norm=True,
    )

    # Distances: (Nc, Na)
    dist = delta_pbc.reshape(Nc, Na)

    # Mask out cross-graph pairs (set to inf so they don't affect min)
    ads_batch = batch[ads_idx]
    same_graph = (cat_batch.unsqueeze(1) == ads_batch.unsqueeze(0))  # (Nc, Na)
    dist[~same_graph] = float('inf')

    # Min distance from each catalyst atom to nearest adsorbate in same graph
    min_dist = dist.min(dim=1).values  # (Nc,)

    # Mark catalyst atoms within cutoff
    mask[cat_idx[min_dist <= cutoff]] = True

    return mask
    