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
            norm: bool=False,
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
        norm: If True, return L1 norm of minimum distance (N, 1)
              If False, return minimum distance vector (N, 3)

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

    # Find nearest lattice translation (integer cell translations)
    n0 = torch.round(u_f)

    if naive:
        # Fast method: just round to nearest integer
        # This gives the approximate minimum image
        u_f_min = u_f - n0
    else:
        # Exact method: search 27 neighboring cells (3^3 combinations)
        # This guarantees finding the true minimum distance

        # Compute metric tensor G = A @ A^T (encodes cell geometry)
        G = A @ A.transpose(1,2)  # (N, 3, 3)

        # Generate all 27 neighboring cell offsets: {-1, 0, 1}^3
        D = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), dtype=torch.int, device=n0.device)  # (27, 3)

        # Test all 27 neighboring cells
        n = n0.unsqueeze(1) + D.unsqueeze(0)  # (N, 1, 3) + (1, 27, 3) = (N, 27, 3)

        # Compute fractional displacement for each candidate
        U_f_n = u_f.unsqueeze(1) - n  # (N, 1, 3) - (N, 27, 3) = (N, 27, 3)

        # Compute squared distance in real space: |u|^2 = u_f^T @ G @ u_f
        W = U_f_n @ G  # (N, 27, 3)
        u_f2 = (W * U_f_n).sum(dim=-1)  # (N, 27) - squared distances

        # Select minimum distance candidate
        k_min = u_f2.argmin(dim=1)  # (N,)
        u_f_min = U_f_n[torch.arange(delta.size(0)), k_min]  # (N, 3)

    # Convert back to Cartesian coordinates: u_min = A @ u_f_min
    u_min = torch.bmm(u_f_min.unsqueeze(1), A).squeeze(1)  # (N, 3)

    if norm:
        # Return L1 norm (mean absolute value across xyz)
        return u_min.abs().mean(dim=1, keepdim=True)  # (N, 1)
    else:
        # Return full minimum image vector
        return u_min

def hungarian(
    s0: Tensor,
    s1: Tensor,
    cell: Tensor,
    batch: Tensor,
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

    # Compute pairwise cost matrix: C[i,j] = distance from s0[i] to s1[j]
    # Using minimum image convention for periodic boundaries
    
    C = s1.unsqueeze(0) - s0.unsqueeze(1)  # (N, N, 3) - all pairwise displacements

    # Apply PBC to get minimum image distances
    C = pbc_aware_delta(
        cell=cell,
        delta=C.reshape(-1, 3),  # Flatten to (N*N, 3) for vectorized computation
        batch=batch,
    ).reshape(s1.size(0), s1.size(0), 3)  # Reshape back to (N, N, 3)

    # Compute L2 distance matrix (cost matrix for Hungarian algorithm)
    C_l2 = torch.linalg.norm(C, dim=-1)  # (N, N)

    # Solve linear assignment problem using Hungarian algorithm
    # Returns row and column indices of optimal assignment
    # row_ind[i] matches to col_ind[i], where col_ind gives permutation of s0
    _, col_ind = assignment_to_indices(cost=C_l2)

    # Apply optimal permutation to s0
    s0_aligned = s0[col_ind]  # (N, 3) - reordered to match s1
    
    s0_aligned = 1 

    return s0_aligned
   
