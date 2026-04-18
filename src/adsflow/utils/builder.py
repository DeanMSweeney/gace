import os
import numpy as np
from ase import Atoms
import ase.io
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import DummySpecie, Element
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Tuple
from adsflow.utils.factorized_builders.data_processor import compute_ads_pairwise_distances

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


# -----------------------------------------------------------------------------
# The following functions are adapted from CatFlow
# (CatFlow/scripts/sampling/assemble.py), authored by the CatFlow team.
# Changes: removed model-output-dict wrapper; accept tensors or numpy arrays;
#          cell matrix input instead of (a,b,c,α,β,γ) tuple;
#          returns ASE Atoms directly.
# -----------------------------------------------------------------------------

def _to_numpy(x):
    """Convert tensor or array-like to numpy array.

    Casts BFloat16/Float16 tensors to Float32 first, since numpy has no
    native bfloat16 dtype (common when the trainer runs bf16-mixed precision).
    """
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def find_surface_atoms_by_height(surface_atoms: Atoms):
    """
    Tag surface vs bulk atoms by height. An atom is considered a surface atom
    if it is within 2 Å of the highest atom along the z-direction (3rd cell vector).

    Returns a list of tags: 1 = surface, 0 = bulk.

    Adapted from CatFlow (scripts/sampling/assemble.py).
    """
    unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
    scaled_positions = surface_atoms.get_scaled_positions()
    scaled_max_height = max(sp[2] for sp in scaled_positions)
    scaled_threshold = scaled_max_height - 2.0 / unit_cell_height

    return [
        0 if sp[2] < scaled_threshold else 1
        for sp in scaled_positions
    ]


def tag_surface_atoms(slab_atoms: Atoms) -> Atoms:
    """
    Return a copy of slab_atoms with surface atoms tagged 1 and bulk atoms tagged 0,
    determined by height along the z-direction.

    Adapted from CatFlow (scripts/sampling/assemble.py).
    """
    slab_atoms = slab_atoms.copy()
    slab_atoms.set_tags(find_surface_atoms_by_height(slab_atoms))
    return slab_atoms


def build_slab_adsorbate(
    prim_pos,
    prim_atomic_numbers,
    lattice_params,
    transformation_matrix,
    ads_pos,
    ads_atomic_numbers,
    vacuum: float = 15.0,
    cell: Optional[np.ndarray] = None,
    prim_slab_atom_mask=None,
    ads_atom_mask=None,
) -> Atoms:
    """
    Build a full slab + adsorbate ASE Atoms object from primitive cell data.

    Pipeline:
        1. Build a pymatgen Structure from the primitive slab
        2. Expand to supercell via make_supercell(transformation_matrix)
        3. Convert to ASE Atoms; scale cell[2] to add vacuum above the slab
        4. Tag slab surface/bulk atoms (1/0); combine with adsorbate (tag 2)

    Args:
        prim_pos:              (N, 3)   Cartesian positions of primitive slab atoms
        lattice_params:        (6,)     (a, b, c, alpha, beta, gamma) — lengths in Å, angles in radians
        transformation_matrix: (3, 3)   Integer supercell matrix (rounded internally)
        ads_pos:               (M, 3)   Adsorbate atom Cartesian positions
        prim_atomic_numbers:   (N,)     Atomic numbers of primitive slab atoms
        ads_atomic_numbers:    (M,)     Atomic numbers of adsorbate atoms
        vacuum:                float    Vacuum thickness in Å to add above the slab (default 15.0)
        prim_slab_atom_mask:   (N,) optional  boolean mask; True = valid atom
        ads_atom_mask:         (M,) optional  boolean mask; True = valid atom

    Returns:
        ASE Atoms object of the full slab + adsorbate system.

    Adapted from CatFlow (scripts/sampling/assemble.py).
    """
    adaptor = AseAtomsAdaptor()

    prim_pos              = _to_numpy(prim_pos)
    lattice_params        = _to_numpy(lattice_params)
    transformation_matrix = _to_numpy(transformation_matrix)
    ads_pos               = _to_numpy(ads_pos)
    prim_atomic_numbers   = _to_numpy(prim_atomic_numbers)
    ads_atomic_numbers    = _to_numpy(ads_atomic_numbers)

    # Apply masks
    if prim_slab_atom_mask is not None:
        mask = _to_numpy(prim_slab_atom_mask).astype(bool)
        prim_pos            = prim_pos[mask]
        prim_atomic_numbers = prim_atomic_numbers[mask]

    if ads_atom_mask is not None:
        ads_mask           = _to_numpy(ads_atom_mask).astype(bool)
        ads_pos            = ads_pos[ads_mask]
        ads_atomic_numbers = ads_atomic_numbers[ads_mask]

    a, b, c, alpha, beta, gamma = lattice_params
    alpha, beta, gamma = np.degrees(alpha), np.degrees(beta), np.degrees(gamma)

    # Build pymatgen Structure from primitive slab
    species = [
        DummySpecie("X") if z == 0 else Element.from_Z(int(z))
        for z in prim_atomic_numbers
    ]
    prim_struct = Structure(
        lattice=Lattice.from_parameters(a, b, c, alpha, beta, gamma),
        species=species,
        coords=prim_pos,
        coords_are_cartesian=True,
    )

    # Expand to supercell
    M = np.round(transformation_matrix).astype(int)
    supercell_struct = prim_struct.copy()
    supercell_struct.make_supercell(M, to_unit_cell=False)

    # Convert to ASE and add vacuum above the slab
    slab = adaptor.get_atoms(supercell_struct)
    cell = slab.get_cell().copy()
    z_height = np.linalg.norm(cell[2])
    cell[2] = cell[2] * (z_height + vacuum) / z_height
    slab.set_cell(cell)

    # Tag slab atoms: 1 = surface, 0 = bulk
    slab = tag_surface_atoms(slab)

    # Build adsorbate Atoms with tag 2
    ads = Atoms(
        numbers=ads_atomic_numbers,
        positions=ads_pos,
        cell=cell,
        pbc=True,
    )
    ads.set_tags(np.full(len(ads), 2, dtype=int))

    return slab + ads


def forge_catalyst(
    opt_graph,
    write_dir: Optional[str] = None,
    batch_idx: Optional[int] = None,
) -> list:
    """
    Build a full slab + adsorbate ASE Atoms object for every sample in a forge batch.

    Reads `*_generated` fields from `opt_graph` (the ODE-sampled output), converts
    fractional coordinates to Cartesian, reconstructs adsorbate positions from COM +
    relative offsets, and calls `build_slab_adsorbate` for each sample.

    Side-effects:
        Attaches two attributes to `opt_graph`:
        - `gen_pos_all`: list of (N_i, 3) numpy arrays — full Cartesian positions
        - `gen_atomic_numbers_all`: list of (N_i,) numpy arrays — atomic numbers

    Args:
        opt_graph:   PyG Batch returned by `sample()`, carrying `*_generated` fields.
        write_dir:   If given, write each structure as a VASP POSCAR to this directory.
        batch_idx:   Batch index used to name POSCAR files (`{batch_idx}_{b}.vasp`).

    Returns:
        List of ASE Atoms objects, one per sample in the batch.
    """
    prim_mask  = (opt_graph.tags == 1)
    ads_mask   = (opt_graph.tags == 2)
    prim_batch = opt_graph.batch[prim_mask]
    ads_batch  = opt_graph.batch[ads_mask]
    # num_graphs is not set on forge-sampled batches; derive from batch index
    num_graphs = int(opt_graph.batch.max().item()) + 1

    atoms_list = []
    gen_pos_all = []
    gen_atomic_numbers_all = []

    for b in range(num_graphs):
        try:
            lengths = opt_graph.lengths_generated[b]   # (3,)
            angles  = opt_graph.angles_generated[b]    # (3,)
            M       = opt_graph.M_generated[b]         # (3, 3)

            cell     = lattice_params_to_matrix(lengths.unsqueeze(0), angles.unsqueeze(0)).squeeze(0)
            frac     = opt_graph.pos_generated[prim_batch == b]           # (N_prim, 3)
            prim_pos = frac @ cell                                         # (N_prim, 3) Cartesian

            ads_rel = opt_graph.ads_rel_pos_generated[ads_batch == b]     # (N_ads, 3)
            ads_pos = opt_graph.ads_com_generated[b].unsqueeze(0) + ads_rel  # (N_ads, 3)

            # atomic_numbers_generated is already 1-indexed (sampler stores argmax + 1)
            # ads_atomic_numbers is 0-indexed (sampler shifts it down by 1 for internal use)
            atoms = build_slab_adsorbate(
                prim_pos=prim_pos,
                lattice_params=torch.cat([lengths, angles]),               # (6,) radians
                transformation_matrix=M,
                ads_pos=ads_pos,
                prim_atomic_numbers=opt_graph.atomic_numbers_generated[prim_batch == b],
                ads_atomic_numbers=opt_graph.ads_atomic_numbers[ads_batch == b] + 1,
                vacuum=15.0,
            )

            atoms_list.append(atoms)
            gen_pos_all.append(atoms.get_positions())
            gen_atomic_numbers_all.append(atoms.get_atomic_numbers())

            # --- DEBUG: write primitive cell before supercell expansion ---
            if write_dir:
                cell_np = _to_numpy(cell)
                prim_pos_np = _to_numpy(prim_pos)
                prim_z_np = _to_numpy(opt_graph.atomic_numbers_generated[prim_batch == b])
                prim_atoms = Atoms(
                    numbers=prim_z_np,
                    positions=prim_pos_np,
                    cell=cell_np,
                    pbc=True,
                )
                os.makedirs(write_dir, exist_ok=True)
                ase.io.write(
                    os.path.join(write_dir, f'{batch_idx}_{b}_prim.vasp'),
                    prim_atoms,
                    format='vasp',
                )
            # --- END DEBUG ---

        except Exception as e:
            print(f"[forge_catalyst] Failed to build graph {b}: {e}")
            atoms_list.append(None)

    opt_graph.gen_pos_all = gen_pos_all
    opt_graph.gen_atomic_numbers_all = gen_atomic_numbers_all

    if write_dir:
        os.makedirs(write_dir, exist_ok=True)
        for b, atoms in enumerate(atoms_list):
            if atoms is not None:
                ase.io.write(
                    os.path.join(write_dir, f'{batch_idx}_{b}.vasp'),
                    atoms,
                    format='vasp',
                )

    return atoms_list


def build_graph_from_atoms(
    mol_atoms: Atoms,
    binding_energy: float,
    num_prim_atoms: int = 3,
) -> Data:
    """Build a minimal PyG Data object for forge generate mode.

    Adsorbate conditioning fields are populated from `mol_atoms`.
    Catalyst fields are zeroed placeholders — _sample_forge reinitialises
    them from the prior distribution.

    Args:
        mol_atoms: ASE Atoms object for the adsorbate molecule.
        binding_energy: Target adsorption energy (eV) used as conditioning signal.
        num_prim_atoms: Number of primitive catalyst atoms to allocate (placeholder).

    Returns:
        PyG Data object with all fields required by _sample_forge.
    """

    ads_pos_np = mol_atoms.get_positions().astype(np.float32)    # (N_ads, 3)
    ads_z_np   = mol_atoms.get_atomic_numbers().astype(np.int64) # (N_ads,)
    N_ads = len(ads_z_np)

    ads_pos_t = torch.from_numpy(ads_pos_np)
    ads_com   = ads_pos_t.mean(dim=0)              # (3,)
    ads_rel_pos = ads_pos_t - ads_com              # (N_ads, 3)
    ads_pairwise_dist = compute_ads_pairwise_distances(ads_pos_t)  # (N_ads, 15)

    tags = torch.cat([
        torch.ones(num_prim_atoms, dtype=torch.long),
        torch.full((N_ads,), 2, dtype=torch.long),
    ])  # (N_prim + N_ads,)

    return Data(
        num_nodes=num_prim_atoms + N_ads,
        pos=torch.zeros(num_prim_atoms, 3),
        tags=tags,
        # Adsorbate conditioning (fixed throughout sampling)
        ads_atomic_numbers=torch.from_numpy(ads_z_np),
        ads_pairwise_dist=ads_pairwise_dist,
        ads_com=ads_com.unsqueeze(0),              # (1, 3) → batches to (B, 3)
        ads_rel_pos=ads_rel_pos,                   # (N_ads, 3)
        ads_binding_energy=torch.tensor(binding_energy, dtype=torch.float32),
        # Placeholder lattice / geometry (overwritten from noise in _sample_forge)
        lengths_scaled=torch.zeros(1, 3),
        angles_radians=torch.zeros(1, 3),
        transformation_matrix=torch.eye(3, dtype=torch.long).unsqueeze(0),  # (1, 3, 3)
    )
