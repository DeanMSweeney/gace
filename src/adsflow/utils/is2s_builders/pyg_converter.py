"""
is2s_builders/converter.py
──────────────────────────
Convert an OC20-style LMDB dict (flat numpy arrays) into a PyG Data object.

This function was previously defined inline in lmdb_loader.py as
dict_to_pyg_data().  It is now centralised here alongside the other
builder utilities so that lmdb_loader stays focused on I/O.

Required dict keys
------------------
    pos             : (N, 3) float array   — Cartesian positions
    cell            : (3, 3) float array   — unit cell
    atomic_numbers  : (N,)   int array
    natoms          : int
    fixed           : (N,)   bool / int array
    tags            : (N,)   int array
    sid             : int
    fid             : int

Optional dict keys (added to Data only when present)
-----------------------------------------------------
    pos_relaxed, velocity           → torch tensors
    Fmax, dE, dSmax, y_init, y_relaxed → scalar tensors
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def dict_to_pyg_data(data_dict: dict) -> Data:
    """
    Convert an OC20-style LMDB dictionary to a PyG Data object.

    Core fields (pos, cell, atomic_numbers, natoms, fixed, tags, sid, fid) are
    always required. Optional fields are added only if present in the dict,
    so both new (raw-structure-only) and old (OC20-dense rich) LMDBs load cleanly.

    Args:
        data_dict: Dictionary with LMDB fields.

    Returns:
        PyG Data object with all available fields as torch tensors.
    """
    # --- Required core fields ---
    kwargs = dict(
        pos=torch.from_numpy(data_dict["pos"]),
        cell=torch.from_numpy(data_dict["cell"]).unsqueeze(0),
        atomic_numbers=torch.from_numpy(data_dict["atomic_numbers"]),
        natoms=torch.tensor(data_dict["natoms"]),
        fixed=torch.from_numpy(np.array(data_dict["fixed"])),
        tags=torch.from_numpy(np.array(data_dict["tags"])),
        sid=data_dict["sid"],
        fid=torch.tensor(data_dict["fid"]),
    )

    # --- Optional fields: added only when present in the stored dict ---
    _optional_numpy = ["pos_relaxed", "velocity"]
    _optional_scalar = ["Fmax", "dE", "dSmax", "y_init", "y_relaxed"]

    for field in _optional_numpy:
        if field in data_dict:
            kwargs[field] = torch.from_numpy(np.array(data_dict[field]))

    for field in _optional_scalar:
        if field in data_dict:
            kwargs[field] = torch.tensor(data_dict[field])

    return Data(**kwargs)
