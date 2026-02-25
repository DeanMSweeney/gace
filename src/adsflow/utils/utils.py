import os
import sys

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

    NOTE: Uses per-graph loop to avoid O(Nc_total × Na_total) memory from
    cross-graph pairs. The vectorized version below is faster but requires
    ~B× more memory (B = batch size). See commented code at end of function.

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

    num_graphs = batch.max().item() + 1
    for g in range(num_graphs):
        g_mask = (batch == g)
        ads_idx = (g_mask & adsorbate).nonzero(as_tuple=True)[0]
        cat_idx = (g_mask & ~adsorbate & not_fixed).nonzero(as_tuple=True)[0]

        if len(ads_idx) == 0 or len(cat_idx) == 0:
            continue

        ads_pos = pos[ads_idx]  # (Na, 3)
        cat_pos = pos[cat_idx]  # (Nc, 3)
        Nc, Na = len(cat_idx), len(ads_idx)

        # Pairwise deltas: (Nc, Na, 3)
        delta = cat_pos.unsqueeze(1) - ads_pos.unsqueeze(0)

        # PBC minimum image distances
        batch_flat = torch.full(
            (Nc * Na,), g, device=pos.device, dtype=torch.long
        )
        dist = pbc_aware_delta(
            cell=cell,
            delta=delta.reshape(-1, 3),
            batch=batch_flat,
            l2_norm=True,
        ).reshape(Nc, Na)

        # Min distance from each catalyst atom to nearest adsorbate
        min_dist = dist.min(dim=1).values  # (Nc,)

        # Mark catalyst atoms within cutoff
        mask[cat_idx[min_dist <= cutoff]] = True

    return mask

class Silent:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Save original file descriptors
        self._original_stdout_fd = os.dup(1)
        self._original_stderr_fd = os.dup(2)

        # Redirect low-level file descriptors
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore low-level file descriptors
        os.dup2(self._original_stdout_fd, 1)
        os.dup2(self._original_stderr_fd, 2)
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)

        # Restore Python streams
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr