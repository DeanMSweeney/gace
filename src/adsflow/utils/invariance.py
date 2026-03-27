"""
Utilities for handling periodic boundary conditions (PBC) and atom permutation invariance.

This module provides functions for:
1. Computing minimum image convention distances (pbc_aware_delta)
2. Finding optimal atom alignments using Hungarian algorithm (hungarian)
"""

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Literal, Union, Tuple
import itertools
from torch_linear_assignment import assignment_to_indices


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

def hungarian(
    set0: Tensor,
    set1: Tensor,
    batch: Optional[Tensor]=None,
    cell: Optional[Tensor]=None,
    pbc_aware: bool=True,
) -> Tensor:
    """
    Find optimal atom permutation between two structures using Hungarian algorithm.

    Given two sets of atomic positions (s0 and s1), finds the permutation of s0
    that minimizes the total distance to s1. This is the solution to the linear
    assignment problem, solved using the Hungarian (Kuhn-Munkres) algorithm.

    Use case: When generating structures from noise (forge variant), catalyst atoms
    need to be permuted to match the target structure's atom ordering.

    Algorithm:
    1. Compute pairwise distance matrix C[i,j] = |s1[j] - s0[i]| (with PBC)
    2. Solve assignment problem: find permutation π minimizing Σ C[i, π(i)]
    3. Return permuted s0 according to optimal assignment

    Args:
        s0: Initial atomic positions (N, 3) - positions to be permuted
        s1: Target atomic positions (N, 3) - reference positions
        cell: Unit cell lattice vectors for PBC handling
        batch: Batch indices for each atom (maps atoms to their structures)

    Returns:
        s0_aligned: Permuted s0 positions (N, 3) optimally aligned to s1
                    Minimizes total displacement: Σ |s1[i] - s0_aligned[i]|

    Example:
        If s0 = [[0,0,0], [1,0,0]] and s1 = [[1,0,0], [0,0,0]],
        returns [[1,0,0], [0,0,0]] (atoms swapped to match s1 ordering)

    Note:
        Both s0 and s1 must have the same number of atoms of the same types.
        Typically called per atom type (e.g., separately for C, O, H atoms).
    """

    if set0.size(0) != set1.size(0):
        raise ValueError("set0 and set1 must have the same number of rows.")

    N = set0.size(0)
    if N == 0:
        return set0

    # Compute pairwise cost matrix: C[i,j] = distance from s0[i] to s1[j]
    C = set1.unsqueeze(0) - set0.unsqueeze(1)  # (N, N, 3)

    # Apply PBC to get minimum image distances
    if pbc_aware:
        # Check if fractional/cartesian coordinates are used
        if set0.max().item() > 1:
            if cell is None:
                raise ValueError("cell must be provided for PBC-aware alignment.")
            if batch is None:
                batch = torch.zeros(N, dtype=torch.long, device=set0.device)
            else:
                batch = batch.long().to(set0.device)
                if batch.size(0) != N:
                    raise ValueError("batch length must match set0/set1 length.")
            # Expand batch indices for all N*N pairwise distances
            # Each row i uses cell from batch[i]
            batch_expanded = batch.unsqueeze(1).expand(-1, N).reshape(-1)  # (N*N,)
            C = pbc_aware_delta(
                delta=C.reshape(-1, 3),  # (N*N, 3)
                cell=cell,
                batch=batch_expanded,
            ).reshape(N, N, 3) # (N, N, 3)
        else:
            # Naive minimum image convention
            Cn0 = torch.round(C)
            C = C - Cn0
  
    # Compute L2 distance matrix (cost matrix for Hungarian algorithm)
    C_l2 = torch.linalg.norm(C, dim=-1)  # (N, N)

    # Solve linear assignment problem using Hungarian algorithm
    sol = linear_assignment(C_l2)
    row_ind, col_ind = assignment_to_indices(sol)

    # Reorder set0 into set1 ordering: aligned[col] = source[row]
    set0_aligned = torch.empty_like(set0)
    set0_aligned[col_ind.long()] = set0[row_ind.long()]

    return set0_aligned
   
