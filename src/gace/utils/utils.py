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


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORARY: flow-trajectory dumping (noise → generated structure).
# Flip DUMP_FLOW_TRAJ to True to write snapshots from Flow.sample(). Off by default.
# ─────────────────────────────────────────────────────────────────────────────
DUMP_FLOW_TRAJ         = True
DUMP_FLOW_TRAJ_DIR     = "flow_traj"
DUMP_FLOW_TRAJ_SAMPLES = 10
DUMP_FLOW_TRAJ_POSCAR  = False   # also write one POSCAR per (sample, t)
DUMP_FLOW_TRAJ_BUILD   = True    # also write the built supercell slab + adsorbate per (sample, t)


def dump_flow_snapshot(
    data: Data,
    step: int,
    every: int,
    t_curr: float,
    n_scale: Tensor,
    force: bool = False,
    out_dir: str = DUMP_FLOW_TRAJ_DIR,
    num_samples: int = DUMP_FLOW_TRAJ_SAMPLES,
    write_poscar: bool = DUMP_FLOW_TRAJ_POSCAR,
    build: bool = DUMP_FLOW_TRAJ_BUILD,
) -> None:
    """
    Dump the current sampling state as viewable structures — debugging aid only.

    Fires every `every` integration steps (and whenever `force` is set, for the
    final t = 1 frame). Each snapshot writes the primitive cell (fractional pos →
    Cartesian via the current lattice) plus the adsorbate at its absolute
    Cartesian ads_pos, tagged 1 = slab, 2 = adsorbate.

    Files (in out_dir), appended frame-by-frame as t advances:
        traj_sample{b}.extxyz  — one file per sample, frames ordered by t
        traj_all.extxyz        — every sample and time in one file
        POSCAR_s{b}_t{t}       — optional, one static POSCAR per frame
        built_sample{b}.extxyz — optional, supercell slab + adsorbate via
                                 build_slab_adsorbate (M rounded, 15 Å vacuum).
                                 Frames that fail to build (e.g. singular M at
                                 small t) are skipped silently.

    extxyz rather than a multi-frame POSCAR because the lattice changes at every
    t and POSCAR/XDATCAR carries a single cell for the whole file.

    Still-masked atom types show up as Fm (Z = 100): the CTMC mask token is
    model.atom_dim - 1 = 99, which un-shifts one past the last real element.

    Args:
        data:        current PyG batch inside the Euler loop (pos = fractional,
                     atomic_numbers = 0-indexed w/ mask token, ads_pos = Å)
        step:        current integration step index
        every:       dump one frame every this many steps
        t_curr:      current flow time (frame label only)
        n_scale:     (B, 1) N^(1/3) length de-normalisation factor
        force:       dump regardless of `step` (used for the final frame)
    """
    if not DUMP_FLOW_TRAJ:
        return

    if not force and step % every != 0:
        return

    from gace.utils.builder import lattice_params_to_matrix, build_slab_adsorbate  # local: temporary code

    os.makedirs(out_dir, exist_ok=True)

    prim_batch = data.batch[data.tags == 1].cpu()
    ads_batch  = data.batch[data.tags == 2].cpu()
    B = int(data.batch.max().item()) + 1

    lengths = (data.lengths_scaled * n_scale).float().cpu()
    angles  = data.angles_radians.float().cpu()
    cells   = lattice_params_to_matrix(lengths=lengths, angles=angles)  # (B, 3, 3)
    M_all   = data.transformation_matrix.float().cpu()
    frac    = data.pos.float().cpu()
    z_prim  = data.atomic_numbers.long().cpu()
    ads_xyz = data.ads_pos.float().cpu()
    z_ads   = data.ads_atomic_numbers.long().cpu()

    # The t = 0 frame truncates any files left over from a previous run
    first = (step == 0)

    # Built frames can fail, so append= can't be used to truncate them; clear up front
    if build and first:
        for f in os.listdir(out_dir):
            if f.startswith("built_") and f.endswith(".extxyz"):
                os.remove(os.path.join(out_dir, f))

    for b in range(min(num_samples, B)):
        cell = cells[b]
        pm, am = (prim_batch == b), (ads_batch == b)

        # 0-indexed → atomic number; the mask token un-shifts to Z = 100 (Fm)
        numbers = torch.cat([z_prim[pm] + 1, z_ads[am] + 1])

        atoms = Atoms(
            numbers=numbers.numpy(),
            positions=torch.cat([frac[pm] @ cell, ads_xyz[am]]).numpy(),
            cell=cell.numpy(),
            pbc=True,
        )
        atoms.set_tags([1] * int(pm.sum()) + [2] * int(am.sum()))
        atoms.info["t"] = round(float(t_curr), 3)

        write(os.path.join(out_dir, f"traj_sample{b:02d}.extxyz"), atoms, format="extxyz", append=not first)
        write(os.path.join(out_dir, "traj_all.extxyz"), atoms, format="extxyz", append=not (first and b == 0))
        if write_poscar:
            write(os.path.join(out_dir, f"POSCAR_s{b:02d}_t{float(t_curr):.2f}"), atoms, format="vasp")

        if build:
            try:
                built = build_slab_adsorbate(
                    prim_pos=frac[pm] @ cell,
                    prim_atomic_numbers=z_prim[pm] + 1,
                    lattice_params=torch.cat([lengths[b], angles[b]]),
                    transformation_matrix=M_all[b],
                    ads_pos=ads_xyz[am],
                    ads_atomic_numbers=z_ads[am] + 1,
                    vacuum=15.0,
                )
                built.info.pop("primitive_cell")  # Atoms object; extxyz can't serialise it
                built.info["t"] = round(float(t_curr), 3)
                write(os.path.join(out_dir, f"built_sample{b:02d}.extxyz"), built, format="extxyz", append=True)
                write(os.path.join(out_dir, "built_all.extxyz"), built, format="extxyz", append=True)
            except Exception:
                pass  # early-t states (e.g. singular rounded M) often can't be built


def center_and_random_so3_rotation(ref_pos: Tensor, ads_batch: Tensor, B: int) -> Tensor:
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
