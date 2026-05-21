"""
catflow_builders/data_processor.py
────────────────────────────────────
Preprocessing utilities applied to the raw CatFlow LMDB dataset before training.

Steps
-----
1. Read each sample dict from an input LMDB file.
2. Extract the primitive cell from ``primitive_slab``.
3. Compute lattice edge lengths and interaxial angles.
4. Compute fractional coordinates from Cartesian positions and the cell matrix.
5. Store all computed fields back into the sample dict.
6. Write the augmented dicts to a new output LMDB file.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import lmdb
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm


# ── Lattice helper ────────────────────────────────────────────────────────────

def cell_to_lengths_and_angles(prim_cell: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute lattice edge lengths and interaxial angles from a cell matrix.

    Args:
        prim_cell: (1, 3, 3) or (3, 3) cell matrix where each row is a
                   lattice vector (a, b, c).

    Returns:
        lengths:        (3,) edge lengths [a, b, c] in Angstrom.
        angles_radians: (3,) interaxial angles [alpha, beta, gamma] in radians,
                        where alpha = angle(b, c), beta = angle(a, c),
                        gamma = angle(a, b).
    """
    cell_matrix = prim_cell.squeeze(0)  # (3, 3)
    a_vec = cell_matrix[0, :]
    b_vec = cell_matrix[1, :]
    c_vec = cell_matrix[2, :]

    a = torch.linalg.norm(a_vec)
    b = torch.linalg.norm(b_vec)
    c = torch.linalg.norm(c_vec)
    lengths = torch.stack([a, b, c])  # (3,)

    cos_alpha = torch.dot(b_vec, c_vec) / (b * c)
    cos_beta  = torch.dot(a_vec, c_vec) / (a * c)
    cos_gamma = torch.dot(a_vec, b_vec) / (a * b)

    alpha = torch.acos(torch.clamp(cos_alpha, -1.0, 1.0))
    beta  = torch.acos(torch.clamp(cos_beta,  -1.0, 1.0))
    gamma = torch.acos(torch.clamp(cos_gamma, -1.0, 1.0))
    angles_radians = torch.stack([alpha, beta, gamma])  # (3,)

    return lengths, angles_radians


# ── Adsorbate pairwise distance helper ───────────────────────────────────────

MAX_ADS_NEIGHBORS = 15  # fixed neighbor dimension for padding — covers all adsorbates in the dataset


def compute_ads_pairwise_distances(ads_pos: Tensor) -> Tensor:
    """Compute pairwise Euclidean distances between all adsorbate atoms,
    including each atom with itself (diagonal = 0), padded to MAX_ADS_NEIGHBORS.

    Padding slots are filled with -1, which sits outside GaussianSmearing's
    [0, d_max] range and produces a near-zero response — distinguishable from
    real distances. The fixed neighbor dimension (MAX_ADS_NEIGHBORS) ensures
    all samples have the same feature width for PyG batching and to_dense_batch.

    Args:
        ads_pos: (N_ads, 3) Cartesian positions in Angstrom.

    Returns:
        dist: (N_ads, MAX_ADS_NEIGHBORS) float32 — real distances in [:N_ads],
              -1 sentinel padding in [N_ads:].
    """
    diff = ads_pos.unsqueeze(1) - ads_pos.unsqueeze(0)       # (N_ads, N_ads, 3)
    dist = torch.linalg.norm(diff, dim=-1)                    # (N_ads, N_ads)

    N_ads = dist.shape[0]
    pad_width = MAX_ADS_NEIGHBORS - N_ads
    if pad_width > 0:
        padding = torch.full((N_ads, pad_width), -1.0)        # (N_ads, pad_width)
        dist = torch.cat([dist, padding], dim=1)               # (N_ads, MAX_ADS_NEIGHBORS)

    return dist  # (N_ads, MAX_ADS_NEIGHBORS)


# ── Fractional coordinate helper ─────────────────────────────────────────────

def cartesian_to_fractional(prim_pos: Tensor, prim_cell: Tensor) -> Tensor:
    """Convert Cartesian positions to fractional coordinates.

    Solves  cell.T @ frac = pos  for each atom.

    Args:
        prim_pos:  (N, 3) Cartesian positions in Angstrom.
        prim_cell: (1, 3, 3) or (3, 3) cell matrix (rows = lattice vectors).

    Returns:
        frac_coords: (N, 3) fractional coordinates in [0, 1).
    """
    N = prim_pos.shape[0]
    cell = prim_cell.squeeze(0)                      # (3, 3)
    cell_bc = cell.unsqueeze(0).expand(N, -1, -1)    # (N, 3, 3)
    frac_coords = torch.linalg.solve(
        cell_bc.transpose(1, 2), prim_pos.unsqueeze(-1)
    ).squeeze(-1)                                    # (N, 3)
    return frac_coords


# ── Per-sample processing ─────────────────────────────────────────────────────

def process_sample(raw: bytes) -> bytes:
    """Augment a single pickled sample dict with lattice params and frac coords.

    Designed to be called from a multiprocessing worker — takes and returns raw
    pickle bytes so no shared state is needed.

    Args:
        raw: Pickled sample dict read directly from the source LMDB.

    Returns:
        Pickled augmented sample dict (protocol 4).
    """
    d = pickle.loads(raw)

    if "primitive_slab" in d:
        slab = d["primitive_slab"]

        prim_cell = torch.from_numpy(
            np.array(slab.get_cell()).astype(np.float32)
        ).unsqueeze(0)  # (1, 3, 3)

        prim_pos = torch.from_numpy(
            slab.get_positions().astype(np.float32)
        )  # (N, 3)

        lengths, angles_radians = cell_to_lengths_and_angles(prim_cell)
        frac_coords = cartesian_to_fractional(prim_pos, prim_cell)

        d["lengths_scaled"] = lengths.numpy()         # (3,)   float32
        d["angles_radians"] = angles_radians.numpy()  # (3,)   float32
        d["frac_coords"]    = frac_coords.numpy()     # (N, 3) float32

    if "ref_ads_pos" in d:
        ref_ads_pos = torch.from_numpy(
            np.array(d["ref_ads_pos"], dtype=np.float32)
        )  # (N_ads, 3)
        ads_pairwise_dist = compute_ads_pairwise_distances(ref_ads_pos)
        d["ads_pairwise_dist"] = ads_pairwise_dist.numpy()  # (N_ads, MAX_ADS_NEIGHBORS) float32

    if "ads_pos" in d:
        ads_pos = torch.from_numpy(
            np.array(d["ads_pos"], dtype=np.float32)
        )  # (N_ads, 3) Cartesian positions
        ads_com     = ads_pos.mean(dim=0)       # (3,)       geometric center
        ads_rel_pos = ads_pos - ads_com         # (N_ads, 3) relative to COM
        d["ads_com"]     = ads_com.numpy()      # (3,)       float32
        d["ads_rel_pos"] = ads_rel_pos.numpy()  # (N_ads, 3) float32

    return pickle.dumps(d, protocol=4)
