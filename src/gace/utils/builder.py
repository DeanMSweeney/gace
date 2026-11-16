import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import DummySpecie, Element
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Tuple
from gace.utils.factorized_builders.data_processor import compute_ads_pairwise_distances

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

    Adapted from CatFlow (Catflow/scripts/sampling/assemble.py).
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

    # Convert primtive strucutre to ase atoms 
    prim = adaptor.get_atoms(prim_struct)

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

    atoms = slab + ads
    atoms.info["primitive_cell"] = prim
    atoms.info["transformation_matrix"] = M

    return atoms


def rebuild_slab_from_prim(atoms: Atoms, vacuum: float) -> Atoms:
    """Rebuild the bare slab from atoms.info's primitive cell and supercell matrix M.

    A .traj round-trip may return the primitive cell as Atoms or as an Atoms.todict() dict.
    """
    prim = atoms.info["primitive_cell"]
    if isinstance(prim, dict):
        prim = Atoms(numbers=np.asarray(prim["numbers"]),
                     positions=np.asarray(prim["positions"]),
                     cell=np.asarray(prim["cell"]),
                     pbc=prim.get("pbc", True))
    elif not isinstance(prim, Atoms):
        raise TypeError(f"primitive_cell is a {type(prim).__name__}, expected Atoms or dict")

    M = np.asarray(atoms.info["transformation_matrix"], dtype=float)
    a, b, c, alpha, beta, gamma = prim.cell.cellpar()
    return build_slab_adsorbate(
        prim_pos=prim.get_positions(),
        prim_atomic_numbers=prim.get_atomic_numbers(),
        lattice_params=np.array([a, b, c, np.radians(alpha), np.radians(beta), np.radians(gamma)]),
        transformation_matrix=M,
        ads_pos=np.empty((0, 3)),
        ads_atomic_numbers=np.empty((0,), dtype=int),
        vacuum=vacuum,
    )


def sample_prim_atom_counts(
    B: int,
    device: torch.device,
    hist: Optional[torch.Tensor] = None,
    num_prim_atoms: Optional[int] = None,
) -> torch.Tensor:
    """Sample per-graph primitive atom counts.

    num_prim_atoms takes priority over hist if both are set.
    """
    if num_prim_atoms is not None:
        return torch.full((B,), num_prim_atoms, dtype=torch.long, device=device)
    return torch.multinomial(hist.expand(B, -1), num_samples=1).squeeze(1)


def build_graph_from_atoms(
    mol_atoms: Atoms,
    num_prim_atoms: int = 3,
) -> Data:
    """Build a minimal PyG Data object for forge generate mode.

    Adsorbate conditioning fields are populated from `mol_atoms`.
    Catalyst fields are zeroed placeholders — _sample_forge reinitialises
    them from the prior distribution. ads_binding_energy is deliberately
    absent: predict_step sets it from sampler_cfg.target_binding_energy.

    Args:
        mol_atoms: ASE Atoms object for the adsorbate molecule.
        num_prim_atoms: Number of primitive catalyst atoms to allocate (placeholder).

    Returns:
        PyG Data object with all fields required by _sample_forge.
    """

    ads_pos_np = mol_atoms.get_positions().astype(np.float32)    # (N_ads, 3)
    ads_z_np   = mol_atoms.get_atomic_numbers().astype(np.int64) # (N_ads,)
    N_ads = len(ads_z_np)

    ads_pos_t = torch.from_numpy(ads_pos_np)
    ads_pairwise_dist = compute_ads_pairwise_distances(ads_pos_t)  # (N_ads, 15)

    tags = torch.cat([
        torch.ones(num_prim_atoms, dtype=torch.long),
        torch.full((N_ads,), 2, dtype=torch.long),
    ])  # (N_prim + N_ads,)

    return Data(
        num_nodes=num_prim_atoms + N_ads,
        pos=torch.zeros(num_prim_atoms, 3),
        tags=tags,
        ads_atomic_numbers=torch.from_numpy(ads_z_np),
        ref_ads_pos=ads_pos_t,                     # (N_ads, 3) — centered on-the-fly in sample()
        ads_pairwise_dist=ads_pairwise_dist,       # (N_ads, 15) — used in validity check
        lengths_scaled=torch.zeros(1, 3),
        angles_radians=torch.zeros(1, 3),
        transformation_matrix=torch.eye(3, dtype=torch.long).unsqueeze(0),  # (1, 3, 3)
    )
