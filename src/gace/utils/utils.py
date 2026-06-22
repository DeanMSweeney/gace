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
from gace.utils.invariance import pbc_aware_delta

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


def random_so3_rotation(ref_pos: Tensor, ads_batch: Tensor, B: int) -> Tensor:
    """Center ref_pos by its per-graph centroid and apply a random SO(3) rotation.

    Rotation matrices are sampled uniformly (Haar measure) via QR decomposition
    of a random Gaussian matrix, with det fixed to +1 (proper rotations only).

    Args:
        ref_pos:   (N_ads, 3) reference adsorbate positions in Cartesian Å.
        ads_batch: (N_ads,) graph index for each adsorbate atom.
        B:         Number of graphs in the batch.

    Returns:
        (N_ads, 3) centered and randomly rotated positions.
    """
    from torch_scatter import scatter

    com = scatter(ref_pos, ads_batch, dim=0, dim_size=B, reduce="mean")  # (B, 3)
    centered = ref_pos - com[ads_batch]                                   # (N_ads, 3)

    M = torch.randn(B, 3, 3, device=ref_pos.device)
    Q, _ = torch.linalg.qr(M)
    sign = torch.linalg.det(Q).sign().unsqueeze(-1).unsqueeze(-1)        # (B, 1, 1)
    R = Q * sign                                                          # (B, 3, 3)

    R_per_atom = R[ads_batch]                                             # (N_ads, 3, 3)
    return torch.bmm(R_per_atom, centered.unsqueeze(-1)).squeeze(-1)     # (N_ads, 3)


def record_cfg_conditioning(
    data: Data,
    pred_cond: dict,
    pred_uncond: dict,
    save_path: str,
) -> dict:
    """Record the per-graph conditional-vs-unconditional prediction gap.

    Diagnostic for "is the binding-energy label being washed out?". For each
    graph we measure how far the conditional model output sits from the
    unconditional (null-embedding) output, per modality, and pair it with that
    graph's binding-energy label. Appends to `save_path` across calls so a sweep
    over many samples builds up aligned vectors:

        d = torch.load(save_path)
        plt.scatter(d["ads_binding_energy"], d["gap_pos"])   # conditioning vs BE

    A gap that stays ~0 across all binding energies is direct evidence the
    conditioning is inert; a gap that tracks/varies with binding energy shows it
    is alive.

    Args:
        data:        current PyG Batch (provides batch, tags, ads_binding_energy)
        pred_cond:   conditional output dict   (pos, z, len, ang, M, ads_pos)
        pred_uncond: unconditional output dict (force_unconditional=True)
        save_path:   .pt file to append diagnostics to

    Returns:
        The per-graph diagnostics dict appended this call (CPU tensors).
    """
    prim_batch = data.batch[data.tags == 1]
    ads_batch  = data.batch[data.tags == 2]
    B = int(data.ads_binding_energy.shape[0])

    def per_atom_gap(a: Tensor, b: Tensor, index: Tensor) -> Tensor:
        # RMS of (cond - uncond) reduced to one scalar per graph
        sq = (a.float() - b.float()).pow(2).flatten(1).mean(-1)           # (n_atoms,)
        total = torch.zeros(B, dtype=sq.dtype, device=sq.device)
        count = torch.zeros(B, dtype=sq.dtype, device=sq.device)
        total.index_add_(0, index, sq)
        count.index_add_(0, index, torch.ones_like(sq))
        return (total / count.clamp(min=1)).sqrt()                       # (B,)

    def graph_gap(a: Tensor, b: Tensor) -> Tensor:
        return (a.float() - b.float()).pow(2).flatten(1).mean(-1).sqrt()  # (B,)

    rec = {
        "ads_binding_energy": data.ads_binding_energy.detach().float().cpu(),
        "gap_pos":     per_atom_gap(pred_cond["pos"],     pred_uncond["pos"],     prim_batch).cpu(),
        "gap_z":       per_atom_gap(pred_cond["z"],       pred_uncond["z"],       prim_batch).cpu(),
        "gap_ads_pos": per_atom_gap(pred_cond["ads_pos"], pred_uncond["ads_pos"], ads_batch).cpu(),
        "gap_len":     graph_gap(pred_cond["len"], pred_uncond["len"]).cpu(),
        "gap_ang":     graph_gap(pred_cond["ang"], pred_uncond["ang"]).cpu(),
        "gap_M":       graph_gap(pred_cond["M"],   pred_uncond["M"]).cpu(),
    }

    # Append to any existing record so a multi-batch sweep accumulates.
    if os.path.exists(save_path):
        prev = torch.load(save_path)
        rec = {k: torch.cat([prev[k], v]) for k, v in rec.items()}

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(rec, save_path)
    return rec
