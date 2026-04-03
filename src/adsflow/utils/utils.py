import os
import sys

import numpy as np 
from ase import Atoms
from ase.io import read, write
from ase.data import atomic_masses

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


def adsorbate_center_of_mass(
    data: Data,
    adsorbate_tag: int = 2,
) -> Tensor:
    """
    Compute adsorbate center of mass (COM) for batched PyG data.

    Args:
        data: PyG Data/Batch with at least `pos`, `tags`, and optional `batch`,
              `atomic_numbers`.
        adsorbate_tag: Value in `data.tags` identifying adsorbate atoms.
        mass_weighted: If True, use atomic masses from `data.atomic_numbers`.
                       If False, returns geometric center of adsorbate atoms.
        expand_to_atoms: If True, return per-atom COM tensor of shape (N, 3),
                         where each atom receives its graph's adsorbate COM.
                         If False, return per-graph COM tensor of shape (B, 3).

    Returns:
        Tensor of shape (B, 3) or (N, 3) depending on `expand_to_atoms`.
    """
    pos = data.pos.float()
    N = pos.size(0)
    if hasattr(data, "batch"):
        batch = data.batch.long()
    else:
        batch = torch.zeros(N, dtype=torch.long, device=pos.device)

    ads_mask = (data.tags == adsorbate_tag)
    if not torch.any(ads_mask):
        raise ValueError(f"No adsorbate atoms found with tag={adsorbate_tag}.")

    graph_ids = batch[ads_mask]
    pos_ads = pos[ads_mask]

    weights = torch.ones((pos_ads.size(0), 1), dtype=pos.dtype, device=pos.device)

    num_graphs = int(batch.max().item()) + 1
    com = torch.zeros((num_graphs, 3), dtype=pos.dtype, device=pos.device)
    weight_sum = torch.zeros((num_graphs, 1), dtype=pos.dtype, device=pos.device)

    com.index_add_(0, graph_ids, pos_ads * weights)
    weight_sum.index_add_(0, graph_ids, weights)

    missing = (weight_sum.squeeze(1) == 0)
    if torch.any(missing):
        missing_graphs = torch.where(missing)[0].tolist()
        raise ValueError(
            f"Missing adsorbate atoms in graph(s): {missing_graphs}. "
            "Each graph must contain at least one adsorbate atom."
        )

    com = com / weight_sum
    return com


def lattice_params_to_matrix(
    lengths: Tensor,
    angles: Tensor,
) -> Tensor:
    """
    Convert lattice parameters (lengths and angles) to lattice matrix.

    Args:
        lengths: (B, 3) or (3,) edge lengths [a, b, c] in Angstroms
        angles: (B, 3) or (3,) angles [alpha, beta, gamma] in radians
                alpha = angle between b and c
                beta = angle between a and c
                gamma = angle between a and b

    Returns:
        cell: (B, 3, 3) or (3, 3) lattice matrix where rows are lattice vectors
    """
    # Handle both batched and unbatched inputs
    if lengths.dim() == 1:
        lengths = lengths.unsqueeze(0)
        angles = angles.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    cos_a, cos_b, cos_g = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
    sin_a, sin_b        = torch.sin(angles[:, 0]), torch.sin(angles[:, 1])

    # gamma_star: intermediate angle (pymatgen convention, numerically stable)
    val = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    # c along z, a in xz-plane, b general
    a_vec = torch.stack([lengths[:, 0] * sin_b,
                         torch.zeros(lengths.shape[0], device=lengths.device, dtype=lengths.dtype),
                         lengths[:, 0] * cos_b], dim=1)
    b_vec = torch.stack([-lengths[:, 1] * sin_a * torch.cos(gamma_star),
                          lengths[:, 1] * sin_a * torch.sin(gamma_star),
                          lengths[:, 1] * cos_a], dim=1)
    c_vec = torch.stack([torch.zeros(lengths.shape[0], device=lengths.device, dtype=lengths.dtype),
                         torch.zeros(lengths.shape[0], device=lengths.device, dtype=lengths.dtype),
                         lengths[:, 2]], dim=1)

    # Stack into matrix (B, 3, 3)
    cell = torch.stack([a_vec, b_vec, c_vec], dim=1)

    if squeeze_output:
        cell = cell.squeeze(0)

    return cell


def tile_supercell(
    pos: Tensor,
    cell: Tensor,
    M: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Tile primitive cell positions into a supercell — pure PyTorch, GPU-native.

    Given N atoms in a primitive cell with lattice vectors `cell` (3×3),
    generates the supercell defined by integer matrix M by enumerating all
    fractional lattice translations inside the parallelepiped of M.

    Args:
        pos:   (N, 3) Cartesian positions of primitive cell atoms
        cell:  (3, 3) primitive cell matrix (rows = lattice vectors)
        M:     (3, 3) integer supercell matrix (will be rounded)

    Returns:
        tiled_pos:   (K*N, 3) Cartesian positions of all supercell atoms
        super_cell:  (3, 3)   supercell lattice matrix = M @ cell
    """
    device = pos.device
    dtype  = pos.dtype

    M_int = torch.round(M).long()
    M_float = M_int.float().to(device)

    # Supercell lattice: L_super = M @ L_prim  (each row is a lattice vector)
    super_cell = M_float @ cell                          # (3, 3)

    # Find all integer lattice points inside the parallelepiped of M.
    # A point n = (n1, n2, n3) is inside if M_inv @ n has all components in [0, 1).
    M_inv = torch.linalg.inv(M_float)                    # (3, 3)

    # Scan range: use |M| to bound the search space
    abs_M = M_int.abs()
    max_range = abs_M.sum(dim=0).max().item() + 1        # conservative bound

    # Generate candidate integer grid
    r = torch.arange(-max_range, max_range + 1, device=device)
    grid = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1)  # (R, R, R, 3)
    candidates = grid.reshape(-1, 3).float()             # (C, 3)

    # Filter: keep points where M_inv @ n is in [0, 1) for all components
    frac = candidates @ M_inv.T                          # (C, 3)
    eps = 1e-6
    inside = ((frac >= -eps) & (frac < 1.0 - eps)).all(dim=-1)  # (C,)
    shifts_frac = candidates[inside]                     # (K, 3)

    # Convert fractional shifts to Cartesian
    shifts_cart = shifts_frac @ cell                      # (K, 3)

    # Tile: each shift generates a copy of all N atoms
    # tiled = pos[None, :, :] + shifts[:, None, :]
    K = shifts_cart.shape[0]
    N = pos.shape[0]
    tiled_pos = pos.unsqueeze(0).expand(K, -1, -1) + shifts_cart.unsqueeze(1)  # (K, N, 3)
    tiled_pos = tiled_pos.reshape(K * N, 3)              # (K*N, 3)

    return tiled_pos, super_cell


def build_slab_adsorbate(
    prim_pos: Tensor,
    prim_cell: Tensor,
    supercell_matrix: Tensor,
    ads_pos: Tensor,
    prim_atomic_numbers: Tensor,
    ads_atomic_numbers: Tensor,
    vacuum: float = 15.0,
) -> dict:
    """
    Build full slab + adsorbate structure from primitive cell data.
    Pure PyTorch — stays on GPU, no pymatgen or CPU round-trip.

    Pipeline:
        1. Tile primitive cell using supercell matrix M
        2. Add vacuum in z direction (fixed, not predicted)
        3. Tile atomic numbers and build tags
        4. Concatenate slab + adsorbate

    Args:
        prim_pos:             (N, 3)   Cartesian positions of primitive slab atoms
        prim_cell:            (3, 3)   Primitive cell matrix
        supercell_matrix:     (3, 3)   Supercell matrix (rounded to int internally)
        ads_pos:              (M, 3)   Adsorbate atom positions
        prim_atomic_numbers:  (N,)     Atomic numbers of primitive slab
        ads_atomic_numbers:   (M,)     Atomic numbers of adsorbate
        vacuum:               float    Vacuum thickness in Å (default 15.0)

    Returns:
        dict with:
            pos:              (N_super + M, 3)
            cell:             (3, 3) with vacuum
            atomic_numbers:   (N_super + M,)
            tags:             (N_super + M,)  1=slab, 2=adsorbate
            natoms:           int
    """
    device = prim_pos.device

    # 1. Tile primitive cell
    slab_pos, super_cell = tile_supercell(prim_pos, prim_cell, supercell_matrix)
    N_prim = prim_pos.shape[0]
    K = slab_pos.shape[0] // N_prim   # number of tiling copies

    # 2. Add vacuum in z
    z_vec = super_cell[2]
    z_height = z_vec.norm()
    if z_height > 0:
        scale = (z_height + vacuum) / z_height
        super_cell = super_cell.clone()
        super_cell[2] = z_vec * scale

    # 3. Tile atomic numbers, build tags
    slab_z = prim_atomic_numbers.repeat(K)                # (K*N,)
    slab_tags = torch.ones(slab_pos.shape[0], dtype=torch.long, device=device)   # tag=1
    ads_tags  = torch.full((ads_pos.shape[0],), 2, dtype=torch.long, device=device)  # tag=2

    # 4. Concatenate
    combined_pos  = torch.cat([slab_pos, ads_pos], dim=0)
    combined_z    = torch.cat([slab_z, ads_atomic_numbers], dim=0)
    combined_tags = torch.cat([slab_tags, ads_tags], dim=0)

    return {
        "pos": combined_pos,               # (N_super + M, 3)
        "cell": super_cell,                # (3, 3)
        "atomic_numbers": combined_z,      # (N_super + M,)
        "tags": combined_tags,             # (N_super + M,)  1=slab, 2=adsorbate
        "natoms": combined_pos.shape[0],   # int
    }
