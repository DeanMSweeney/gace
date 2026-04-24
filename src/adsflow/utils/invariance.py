"""
Utilities for handling periodic boundary conditions (PBC) and atom permutation invariance.

This module provides functions for:
1. Computing minimum image convention distances (pbc_aware_delta)
2. Per-graph optimal transport alignment of sparse atom tensors (batched_ot_align)
"""

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import to_dense_batch
from typing import Optional
from torch_linear_assignment import batch_linear_assignment


def pbc_aware_delta(
            cell: Tensor,
            delta: Tensor,
            batch: Optional[Tensor]=None,
            naive: bool=True,
            l1_norm: bool=False,
            l2_norm: bool=False,
            cartesian: bool=True,
            cartesian_to_fractional: bool=False, 
) -> Tensor:
    """
    Compute minimum image convention distance under periodic boundary conditions.

    For crystalline structures with periodic boundaries, there are infinite periodic
    images of each atom. This function finds the shortest distance vector between
    two points by considering all periodic images.

    Algorithm:
    1. Convert displacement vector to fractional coordinates: u_f = A^-T @ u
    2. Find nearest periodic image by rounding to nearest lattice translation
    3. Optionally search 27 neighboring cells for true minimum (naive=False)
    4. Convert back to Cartesian coordinates

    Args:
        cell: Unit cell lattice vectors
              Shape: (batch_size, 3, 3) or (3, 3)
              Each row is a lattice vector: [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]
        delta: Displacement vectors in Cartesian coordinates (N, 3)
               Raw difference: delta = pos_b - pos_a
        batch: Batch indices for each atom (N,) - maps atoms to their cell
               If None, assumes single structure
        naive: If True, use simple rounding (fast but approximate)
               If False, search 27 neighboring cells (exact but slower)
        l1_norm: If True, return L1 norm (mean absolute value across xyz) (N, 1)
        l2_norm: If True, return L2 norm (Euclidean distance) (N, 1)

    Returns:
        u_min: Minimum image distance vector (N, 3) or norm (N, 1)
               This is the shortest distance considering all periodic images

    Example:
        For a cubic cell with side length L, if delta = [0.9*L, 0, 0],
        the minimum image is [-0.1*L, 0, 0] (wrapping around the boundary)
    """

    # Convert inputs to tensors if needed
    if not isinstance(cell, Tensor):
        cell = torch.from_numpy(np.asarray(cell)).to(dtype=torch.float32)
    if not isinstance(delta, Tensor):
        delta = torch.from_numpy(np.asarray(delta)).to(dtype=torch.float32)

    # Broadcast cell to per-atom shape (N, 3, 3)
    N = torch.zeros(delta.size(0), dtype=torch.long)  # indices for broadcasting
    A = cell.unsqueeze(0)[N] if batch is None else cell[batch]

    # Convert displacement to fractional coordinates
    # A^T @ u_f = u  =>  u_f = A^-T @ u
    u = delta  # (N, 3)
    u_f = torch.linalg.solve(A.transpose(1,2), u.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    if cartesian_to_fractional:
        return u_f

    # Find nearest lattice translation (integer cell translations)
    n0 = torch.round(u_f)
    # Fast method: just round to nearest integer
    # This gives the approximate minimum image
    u_f_mic = u_f - n0 

    # Convert back to Cartesian coordinates: u_min = A @ u_f_min
    if cartesian: 
        u_mic = torch.bmm(u_f_mic.unsqueeze(1), A).squeeze(1)  # (N, 3)
    else:
        u_mic = u_f_mic

    if l1_norm:
        # Return L1 norm (mean absolute value across xyz)
        return u_mic.abs().mean(dim=1, keepdim=True)  # (N, 1)
    elif l2_norm:
        # Return Euclidean distance (L2 norm)
        return u_mic.norm(dim=1, keepdim=True)  # (N, 1)
    else:
        # Return full minimum image vector
        return u_mic

def batched_ot_align(
    source: Tensor,
    target: Tensor,
    batch: Tensor,
    pbc_aware: bool = False,
    return_assignment: bool = False,
) -> Tensor:
    """
    Per-graph optimal transport alignment of sparse atom tensors.

    For each graph b, finds the permutation of source atoms that minimises total
    L2 transport cost to target atoms. Target (data) is never modified — only
    the source (noise) is permuted. This ensures x1 remains an
    unmodified data draw at both train and inference time.

    Args:
        source:    (N_total, D) noise tensor — permuted to align with target
        target:    (N_total, D) true structure tensor — used as reference, never permuted
        batch:     (N_total,) graph index for each atom
        pbc_aware: if True, applies minimum-image convention before computing
                   distances (use for fractional coordinates)

    Returns:
        source_aligned: (N_total, D) source with rows permuted per-graph so that
                        source_aligned[i] ≈ target[i] for every atom i
    """
    # Densify sparse atoms → (B, N_max, D) with boolean mask (True = real atom)
    target_dense, mask = to_dense_batch(target, batch)  # (B, N_max, D), (B, N_max)
    source_dense, _    = to_dense_batch(source, batch)  # (B, N_max, D)
    B, N_max, D = target_dense.shape

    # diff[b, i, j] = source[b, j] - target[b, i]
    # cost[b, i, j] = distance from target atom i to source atom j
    diff = source_dense.unsqueeze(1) - target_dense.unsqueeze(2)  # (B, N_max, N_max, D)

    if pbc_aware:
        diff = diff - torch.round(diff)  # minimum image in fractional coords

    cost = torch.linalg.norm(diff, dim=-1)  # (B, N_max, N_max)

    # Exclude padding slots from the assignment
    INF = 1e9
    pad = ~mask  # (B, N_max)
    cost = cost.masked_fill(pad.unsqueeze(2), INF)  # target padding rows
    cost = cost.masked_fill(pad.unsqueeze(1), INF)  # source padding cols

    # Solve B independent assignment problems
    # assignment[b, i] = j  →  target atom i is matched to source atom j
    assignment = batch_linear_assignment(cost)  # (B, N_max)

    # Gather source into aligned positions:
    #   source_aligned[b, i, :] = source[b, assignment[b,i], :]
    # → source_aligned[i] holds the source atom matched to target[i]
    # → source_aligned[i] ≈ target[i]
    idx = assignment.unsqueeze(-1).expand(B, N_max, D)
    source_aligned_dense = torch.gather(source_dense, 1, idx)

    # Return sparse (N_total, D), optionally also return the (B, N_max) assignment
    if return_assignment:
        return source_aligned_dense[mask], assignment
    return source_aligned_dense[mask]
   
