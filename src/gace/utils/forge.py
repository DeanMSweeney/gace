import csv
import logging
import os
import random
import time
import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from gace.utils.builder import build_slab_adsorbate, lattice_params_to_matrix
from gace.utils.relaxation import relax_w_mlff

_CSV_DECIMALS = 6
_FMT = f".{_CSV_DECIMALS}f"
_BINDING_E_THRESHOLD = 0.10  # eV
_THRESHOLD_MULTIPLIERS = [1, 2, 3, 5, 10]

def _csv_fmt(v) -> str:
    if isinstance(v, float):
        return format(v, _FMT)
    return str(v) if v is not None else ""


def _threshold_flags(diff: float | None) -> dict:
    """Return within_NT booleans for each multiplier of _BINDING_E_THRESHOLD."""
    if diff is None:
        return {f"within_{m}T": None for m in _THRESHOLD_MULTIPLIERS}
    return {f"within_{m}T": float(diff <= m * _BINDING_E_THRESHOLD) for m in _THRESHOLD_MULTIPLIERS}


_THRESHOLD_SUBDIRS = {1: "1_threshold", 2: "2_threshold", 3: "3_threshold", 5: "5_threshold", 10: "10_threshold"}


def _threshold_subdir(diff: float | None) -> str:
    """Return the tightest threshold subdirectory name, or 'inf_threshold' if outside all."""
    if diff is not None:
        for m in _THRESHOLD_MULTIPLIERS:
            if diff <= m * _BINDING_E_THRESHOLD:
                return _THRESHOLD_SUBDIRS[m]
    return "inf_threshold"

class CatalystForge:
    """Build, sanitize, relax, and write generated catalyst structures.

    Two modes controlled by what is provided at construction:
        OTF        — opt_graph is set; builds structures from model output batch.
        Preloaded  — preloaded_atoms is set; dict[path, Atoms] supplied by caller.

    Usage (OTF):
        cf = CatalystForge(opt_graph=opt_graph, binding_energies=..., batch_idx=..., ...)
        metrics = cf.fire()

    Usage (post-processing):
        preloaded = load_atoms_parallel(files, num_workers)
        cf = CatalystForge(preloaded_atoms=preloaded, gen_POS="./gen_strucs", ...)
        rows = cf.fire()
    """

    def __init__(
        self,
        batch_idx: int,
        calculator,
        catalyst_dist_cutoff: float,
        generated_strucs: str | None,
        relaxed_strucs: str | None,
        sanitized_strucs: str | None,
        opt_graph=None,
        binding_energies: list[float] | None = None,
        preloaded_atoms: dict[str, "Atoms | None"] | None = None,
        file_ext: str = ".vasp",
        energy_cutoff: float = 10.0,
        sanitize_otf: bool = True,
        relax_otf: bool = False,
        dedupe_otf: bool = False,
        gen_slabs_memory: list | None = None,
        n_strucs: int | None = None,
        seed: int = 42,
        fmax: float = 0.05,
        steps: int = 200,
    ):
        self.opt_graph = opt_graph
        self.preloaded_atoms = preloaded_atoms
        self.binding_energies = binding_energies
        self.batch_idx = batch_idx
        self.calculator = calculator
        self.relax_otf = relax_otf
        self.catalyst_dist_cutoff = catalyst_dist_cutoff
        self.generated_strucs = generated_strucs
        self.relaxed_strucs = relaxed_strucs
        self.sanitized_strucs = sanitized_strucs
        self.file_ext = file_ext
        self.energy_cutoff = energy_cutoff
        self.sanitize_otf = sanitize_otf
        self.n_strucs = n_strucs
        self.seed = seed
        self.fmax = fmax
        self.steps = steps
        self.dedupe_otf = dedupe_otf

        self._all_atoms: list[Atoms | None] = []
        # Running record of seen slabs for OTF dedup. Accept a shared list so the
        # caller can persist it across batches; otherwise keep a local record.
        self.gen_slabs_memory: list = gen_slabs_memory if gen_slabs_memory is not None else []
        # Count of valid structures passed to the OTF dedup check (the denominator
        # for uniqueness; excludes structures that failed to build or sanitize).
        self.num_slabs_checked: int = 0

    def _construct_atoms_objects(self) -> list["Atoms | None"]:
        prim_batch = self.opt_graph.batch[self.opt_graph.tags == 1]
        ads_batch  = self.opt_graph.batch[self.opt_graph.tags == 2]
        num_graphs = int(self.opt_graph.batch.max().item()) + 1
        items = []
        for b in range(num_graphs):
            atoms = self.build_from_factorized_repr(b, prim_batch, ads_batch)
            if atoms is not None:
                atoms.info["fname"]             = f"{self.batch_idx}_{b}{self.file_ext}"
                atoms.info["target_binding_E"]  = self.binding_energies[b]
                sid                             = getattr(self.opt_graph, "sid", None)
                atoms.info["sid"]               = sid[b].item() if sid is not None else ""
            items.append(atoms)
        return items

    def _list_atoms_objects(self) -> list["Atoms | None"]:
        files = sorted(self.preloaded_atoms.keys())
        n = len(files) if self.n_strucs is None else min(self.n_strucs, len(files))
        sample = sorted(random.Random(self.seed).sample(files, n))
        logging.info(f"Processing {len(sample)} structures (preloaded, seed={self.seed})")
        return [self.preloaded_atoms.get(path) for path in sample]

    def build_from_factorized_repr(self, b: int, prim_batch, ads_batch) -> Atoms | None:
        try:
            lengths  = self.opt_graph.lengths_generated[b]
            angles   = self.opt_graph.angles_generated[b]
            M        = self.opt_graph.M_generated[b]
            cell     = lattice_params_to_matrix(lengths.unsqueeze(0), angles.unsqueeze(0)).squeeze(0)
            prim_pos = self.opt_graph.pos_generated[prim_batch == b] @ cell
            ads_pos  = self.opt_graph.ads_pos_generated[ads_batch == b]
            atoms = build_slab_adsorbate(
                prim_pos=prim_pos,
                lattice_params=torch.cat([lengths, angles]),
                transformation_matrix=M,
                ads_pos=ads_pos,
                prim_atomic_numbers=self.opt_graph.atomic_numbers_generated[prim_batch == b],
                ads_atomic_numbers=self.opt_graph.ads_atomic_numbers[ads_batch == b] + 1,
                vacuum=15.0,
            )
            atoms.set_constraint(FixAtoms(atoms.get_tags() == 0))
            return atoms
        except Exception as e:
            logging.warning(f"[build] graph {b}: {e}")
            return None

    def sanitize(self, atoms: Atoms | None) -> None:

        if atoms is None:
            return

        try:
            vol_ok = float(atoms.get_volume()) >= 0.1

            tags = atoms.get_tags()
            cat_mask = tags != 2
            ads_mask = tags == 2

            cat_atoms = atoms[cat_mask]
            ads_idx = np.where(ads_mask)[0]
            cat_idx = np.where(cat_mask)[0]

            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)

            ads_dists = dists[np.ix_(ads_idx, ads_idx)]
            ads_dist_ok = bool(ads_dists.min() >= 0.5)

            cat_dists = dists[np.ix_(cat_idx, cat_idx)]
            cat_atomic_numbers = cat_atoms.get_atomic_numbers()
            atomic_radii = covalent_radii[cat_atomic_numbers]
            sum_atomic_radii = atomic_radii[:, None] + atomic_radii[None, :]
            sum_atomic_radii[sum_atomic_radii == 0.0] = np.inf
            cat_dist_ok = bool((cat_dists >= self.catalyst_dist_cutoff * sum_atomic_radii).all())
            cat_dist_ok_naive = bool(cat_dists.min() >= 0.5)

            cat_oc20_atom_types = np.arange(22, 84)
            cat_atom_types = np.unique(cat_atomic_numbers)
            cat_atom_types_ok = bool(np.isin(cat_atom_types, cat_oc20_atom_types).any())

            cat_ads_dists = dists[cat_mask, :]
            cat_ads_dist_ok = bool(cat_ads_dists.min() >= 0.5)

        except Exception as e:
            vol_ok = False
            cat_dist_ok = False
            cat_dist_ok_naive = False
            cat_atom_types_ok = False
            ads_dist_ok = False
            cat_ads_dist_ok = False

        atoms.info["is_valid"]       = vol_ok and cat_dist_ok and ads_dist_ok and cat_ads_dist_ok and cat_atom_types_ok
        atoms.info["is_valid_base"]  = cat_dist_ok_naive and ads_dist_ok and cat_ads_dist_ok and vol_ok

    def relax(self, atoms: Atoms | None) -> Atoms | None:

        atoms_rlxd, converged = relax_w_mlff(
            atoms=atoms,
            calculator=self.calculator,
            fmax=self.fmax,
            steps=self.steps,
            optimizer="BFGS",
        )

        if not converged: 
            atoms_rlxd = None

        target_E = atoms.info.get("target_binding_E")
        try:
            computed_E = atoms_rlxd.get_potential_energy()
            if computed_E <= self.energy_cutoff:
                atoms_rlxd.info["computed_binding_E"] = computed_E
                if target_E is not None:
                    atoms_rlxd.info["binding_E_diff"] = abs(computed_E - target_E)
                return atoms_rlxd
            atoms.info["is_valid"] = False # binding energy greater than 10 eV, discard sample 
            return None
        except Exception as e:
            logging.warning(f"Energy extraction failed for {atoms.info.get('fname')}: {e}")
            atoms.info["is_valid"] = False # optimization or energy extraction failed, mark as invalid 
            return None
    
    def purge_duplicates_otf(self, atoms) -> bool:
        """OTF duplicate check against the running record of seen slabs.

        Checks the slab (tags != 2) against self.gen_slabs_memory, appending it
        if unique. Returns True if unique (write), False if duplicate (skip).

        The vacuum-padded slab is reduced once up front (in-plane primitive;
        the vacuum keeps the facet/termination intact) and gen_slabs_memory
        stores the reduced form, so every fit() compares already-reduced cells
        (skip_structure_reduction=True) instead of re-reducing the supercell on
        each comparison. Reduction is the dominant cost of fit(), so caching it
        turns the per-comparison cost from O(supercell) to O(primitive).
        """
        self.num_slabs_checked += 1

        slab = atoms #[atoms.get_tags() != 2]
        slab.center(vacuum=15, axis=2)
        pymg_struc = AseAtomsAdaptor().get_structure(slab)
        
        # reduce the slab to its primitive cell 
        matcher = StructureMatcher()
        reduced_pymg_struc = matcher._get_reduced_structure(pymg_struc, matcher._primitive_cell, niggli=True)
        is_unique = True
        for seen_pymg_struc in self.gen_slabs_memory:
            if matcher.fit(seen_pymg_struc, reduced_pymg_struc, skip_structure_reduction=True): 
                is_unique = False
                break

        atoms.info["is_unique"] = is_unique
        if is_unique:
            self.gen_slabs_memory.append(reduced_pymg_struc)
        return is_unique

    def _process_built_struc(
        self,
        atoms: Atoms | None,
        fname: str | None,
        target_E: float | None,
    ) -> dict:
        threshold_cols = {f"within_{m}T": None for m in _THRESHOLD_MULTIPLIERS}
        base_row = {
            "fname":              fname,
            "target_binding_E":   target_E,
            "computed_binding_E": None,
            "binding_E_diff":     None,
            **threshold_cols,
        }

        if not self.sanitize_otf:
            if atoms is not None and self.opt_graph:
                ase.io.write(os.path.join(self.generated_strucs, fname), atoms) # if opt_grpah present, we are generating strucs for the first time
            opt_conditions = self.relax_otf and atoms
            atoms_rlxd = self.relax(atoms=atoms) if opt_conditions else None

            if atoms_rlxd is not None:
                diff = atoms_rlxd.info.get("binding_E_diff")
                subdir = _threshold_subdir(diff)
                atoms_rlxd.info["threshold_subdir"] = subdir
                out_path = os.path.join(self.relaxed_strucs, subdir, fname)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                ase.io.write(out_path, atoms_rlxd)
                return {
                    **base_row,
                    "computed_binding_E": atoms_rlxd.info.get("computed_binding_E"),
                    "binding_E_diff":     diff,
                    **_threshold_flags(diff),
                }
            return base_row

        # sanitize the atoms objects to purge unstabele strucs
        self.sanitize(atoms=atoms)

        # Only dedup validly-built structures so poor builds never enter the
        # uniqueness memory or its denominator.
        if self.dedupe_otf and atoms is not None and atoms.info.get("is_valid"):
            is_unique = self.purge_duplicates_otf(atoms=atoms)
            if not is_unique: # set non-unique strucutres as invalid strucutres
                atoms.info["is_valid"] = False
        else:
            is_unique = True

        # write sanitized and relaxed structures to disk
        if atoms is not None and atoms.info.get("is_valid"):
            if self.relax_otf: 
                atoms_rlxd = self.relax(atoms=atoms) 
            else: 
                atoms_rlxd = None
                ase.io.write(os.path.join(self.sanitized_strucs, fname), atoms)
        else:
            atoms_rlxd = None # if atoms DNE, store atoms_rlxd as None

        if atoms_rlxd is not None:
            diff = atoms_rlxd.info.get("binding_E_diff")
            subdir = _threshold_subdir(diff)
            atoms_rlxd.info["threshold_subdir"] = subdir
            out_path = os.path.join(self.relaxed_strucs, subdir, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if not self.dedupe_otf:
                ase.io.write(out_path, atoms_rlxd) 
            else: 
                if is_unique: 
                    ase.io.write(out_path, atoms_rlxd)
        else:
            diff = None

        validity_keys = ["is_valid", "is_valid_base"]
        return {
            **base_row,
            "computed_binding_E": atoms_rlxd.info.get("computed_binding_E") if atoms_rlxd is not None else None,
            "binding_E_diff":     diff,
            **_threshold_flags(diff),
            **{k: float(atoms.info.get(k)) if atoms is not None else float(False) for k in validity_keys}, # if atoms DNE, it could not be built and mark as invalid
             "is_unique": float(is_unique) if atoms is not None else float(False),
        }

    def fire(self) -> list[dict]:
        """Process structures from opt_graph (OTF) or from preloaded files.

        Returns:
            List of metric/row dicts, one per structure.
        """
        if self.opt_graph is not None:
            items = self._construct_atoms_objects()
            if self.relax_otf: 
                os.makedirs(self.relaxed_strucs, exist_ok=True)
            else:
                os.makedirs(self.sanitized_strucs if self.sanitize_otf else self.generated_strucs, exist_ok=True)
        else:
            items = self._list_atoms_objects()
            os.makedirs(self.sanitized_strucs, exist_ok=True)

        csv_path = None
        write_header = False
        if self.relax_otf:
            os.makedirs(self.relaxed_strucs, exist_ok=True)
            csv_path = os.path.join(self.relaxed_strucs, "relaxed_strucs_data.csv")
            if self.opt_graph is not None and self.batch_idx == 0 and os.path.isfile(csv_path):
                os.remove(csv_path)
            write_header = not os.path.isfile(csv_path)

        atoms_dict_list: list[dict] = []
        csv_fieldnames: list[str] | None = None

        for atoms in tqdm(items, desc="Processing", disable=self.opt_graph is not None):
            fname    = atoms.info.get("fname") if atoms is not None else None
            target_E = atoms.info.get("target_binding_E") if atoms is not None else None
            atoms_dict = self._process_built_struc(atoms=atoms, fname=fname, target_E=target_E)
            atoms_dict_list.append(atoms_dict)
            self._all_atoms.append(atoms)

            if csv_path is not None:
                if csv_fieldnames is None:
                    csv_fieldnames = list(atoms_dict.keys())
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    if write_header:
                        w.writeheader()
                        write_header = False
                    w.writerow({k: _csv_fmt(atoms_dict.get(k)) for k in csv_fieldnames})

        return atoms_dict_list
