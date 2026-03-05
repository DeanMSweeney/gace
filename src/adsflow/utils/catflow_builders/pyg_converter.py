"""
catflow_builders/converter.py
─────────────────────────────
Convert a raw CatFlow LMDB dict (containing an ASE MSONAtoms primitive_slab)
into a PyTorch Geometric Data object.

Slab and adsorbate data are kept separate — they are NOT concatenated.
Slab fields are stored under prim_* names; adsorbate fields are stored
under ads_* names exactly as they appear in the source dict.

Example source dict keys
------------------------
    primitive_slab      : MSONAtoms  (ASE-compatible Atoms object)
    supercell_matrix    : (3, 3) int array
    n_slab              : int
    n_vac               : int
    ads_atomic_numbers  : (N_ads,) int array
    ads_pos             : (N_ads, 3) float array   — Cartesian, Angstrom
    ref_energy          : float
    ref_ads_pos         : (N_ads, 3) float array   — fractional coordinates
    bind_ads_atom       : int   (atomic number of binding atom)
    sid                 : int

Emitted Data fields
-------------------
    prim_pos            : (N_slab, 3)  float32 Cartesian positions
    prim_cell           : (1, 3, 3)   float32 unit cell
    prim_atomic_numbers : (N_slab,)   int64
    prim_natoms         : scalar int
    ads_pos             : (N_ads, 3)  float32 Cartesian positions
    ads_atomic_numbers  : (N_ads,)    int64
    ref_energy          : scalar float
    ref_ads_pos         : (N_ads, 3)  float32 fractional reference positions
    bind_ads_atom       : scalar int
    supercell_matrix    : (3, 3)      int64
    n_slab              : scalar int
    n_vac               : scalar int
    sid                 : int
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def catflow_dict_to_pyg_data(d: dict) -> Data:
    """
    Convert a CatFlow LMDB sample dictionary to a PyG Data object.

    The ASE MSONAtoms primitive_slab is unpacked; each extracted field is
    stored under a ``prim_`` prefix. Adsorbate arrays are stored as-is under
    their original ``ads_*`` names. The two groups are **not** concatenated.

    Args:
        d: CatFlow LMDB sample dictionary with at minimum ``primitive_slab``,
           ``ads_pos``, ``ads_atomic_numbers``, ``ref_energy``,
           ``ref_ads_pos``, ``bind_ads_atom``, and ``sid``.

    Returns:
        PyG ``Data`` object with prim_* and ads_* fields.
    """
    slab = d["primitive_slab"]  # ASE Atoms-compatible (MSONAtoms)

    # ── Slab fields (prim_*) ─────────────────────────────────────────────────
    prim_pos = torch.from_numpy(
        slab.get_positions().astype(np.float32)
    )  # (N_slab, 3)

    prim_cell = torch.from_numpy(
        np.array(slab.get_cell()).astype(np.float32)
    ).unsqueeze(0)  # (1, 3, 3)

    prim_atomic_numbers = torch.from_numpy(
        slab.get_atomic_numbers().astype(np.int64)
    )  # (N_slab,)

    prim_natoms = len(slab)  # scalar int

    # ── Adsorbate fields (ads_*) ─────────────────────────────────────────────
    ads_pos = torch.from_numpy(
        np.array(d["ads_pos"], dtype=np.float32)
    )  # (N_ads, 3)

    ads_atomic_numbers = torch.from_numpy(
        np.array(d["ads_atomic_numbers"], dtype=np.int64)
    )  # (N_ads,)

    # ── Reference / metadata ─────────────────────────────────────────────────
    ref_ads_pos = torch.from_numpy(
        np.array(d["ref_ads_pos"], dtype=np.float32)
    )  # (N_ads, 3) — fractional coordinates

    ref_energy = torch.tensor(float(d["ref_energy"]))

    bind_ads_atom = torch.tensor(int(d["bind_ads_atom"]))

    sid = int(d["sid"])

    # ── Slab geometry metadata ────────────────────────────────────────────────
    supercell_matrix = torch.from_numpy(
        np.array(d["supercell_matrix"], dtype=np.int64)
    )  # (3, 3)

    n_slab = int(d["n_slab"])
    n_vac  = int(d["n_vac"])

    # ── Combined pos + tags (OC20-style) ────────────────────────────────────
    # Concatenate: primitive slab atoms first, adsorbate atoms second.
    # Tags: 1 = slab atom, 2 = adsorbate atom  (matches OC20 convention)
    pos_combined = torch.cat([prim_pos, ads_pos], dim=0)            # (N_slab + N_ads, 3)
    atomic_numbers = torch.cat(
        [prim_atomic_numbers, ads_atomic_numbers], dim=0
    )                                                               # (N_slab + N_ads,)
    tags = torch.cat([
        torch.ones(prim_natoms, dtype=torch.long),                 # slab  → 1
        torch.full((len(ads_atomic_numbers),), 2, dtype=torch.long),  # ads → 2
    ], dim=0)                                                       # (N_slab + N_ads,)

    return Data(
        # Combined (primary fields used by EquiformerV2 / x_matching)
        pos_combined=pos_combined,
        atomic_numbers=atomic_numbers,
        tags=tags,
        # Slab (kept separate for CatFlow-specific flow matching)
        prim_pos=prim_pos,
        prim_cell=prim_cell,
        prim_atomic_numbers=prim_atomic_numbers,
        prim_natoms=prim_natoms,
        # Adsorbate
        ads_pos=ads_pos,
        ads_atomic_numbers=ads_atomic_numbers,
        # Reference / metadata
        ref_ads_pos=ref_ads_pos,
        ref_energy=ref_energy,
        bind_ads_atom=bind_ads_atom,
        # Geometry metadata
        supercell_matrix=supercell_matrix,
        n_slab=n_slab,
        n_vac=n_vac,
        sid=sid,
    )
