import csv
import glob
import logging
import os
import random

import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from tqdm import tqdm

from gace.utils.builder import build_slab_adsorbate, lattice_params_to_matrix
from gace.utils.relaxation import _fmt, relax_w_mlff

_initialized_csvs: set = set()


class CatalystForge:
    """Build, sanitize, relax, and write generated catalyst structures.

    Two modes controlled by whether opt_graph is provided:
        OTF  — opt_graph is set; builds structures from model output batch.
        Disk — opt_graph is None; globs structures from gen_POS directory.

    Usage (OTF):
        cf = CatalystForge(opt_graph=opt_graph, binding_energies=..., batch_idx=..., ...)
        metrics = cf.run()

    Usage (post-processing):
        cf = CatalystForge(gen_POS="./gen_strucs", ...)
        rows = cf.run()
    """

    def __init__(
        self,
        binding_energies: list[float],
        batch_idx: int,
        calculator,
        catalyst_dist_cutoff: float,
        gen_POS: str | None,
        rlxd_POS: str | None,
        opt_graph=None,
        file_ext: str = ".vasp",
        energy_cutoff: float = 10.0,
        ads_integrity_threshold: float = 2.0,
        sanitize_otf: bool = True,
        relax_otf: bool = False,
        config_vasp: bool = False,
        n_strucs: int | None = None,
        seed: int = 42,
    ):
        self.opt_graph = opt_graph
        self.binding_energies = binding_energies
        self.batch_idx = batch_idx
        self.calculator = calculator
        self.relax_otf = relax_otf
        self.config_vasp = config_vasp
        self.catalyst_dist_cutoff = catalyst_dist_cutoff
        self.gen_POS = gen_POS
        self.rlxd_POS = rlxd_POS
        self.file_ext = file_ext
        self.energy_cutoff = energy_cutoff
        # self.ads_integrity_threshold = ads_integrity_threshold
        self.sanitize_otf = sanitize_otf
        self.n_strucs = n_strucs
        self.seed = seed

        self._all_atoms: list[Atoms | None] = []

    def run(self) -> list[dict]:
        """Process structures from opt_graph (OTF) or from gen_POS directory (post-processing).

        Returns:
            List of metric/row dicts, one per structure.
        """
        if self.opt_graph is not None:
            items = self._items_from_graph()
        else:
            items = self._items_from_files()

        os.makedirs(self.gen_POS, exist_ok=True)
        if self.relax_otf:
            os.makedirs(self.rlxd_POS, exist_ok=True)

        # OTF: set up append-mode CSV for binding energies
        otf_csv_path = None
        otf_first_write = False
        if self.opt_graph is not None:
            csv_dir = self.rlxd_POS if self.relax_otf else self.gen_POS
            otf_csv_path = os.path.join(csv_dir, "binding_energies.csv")
            otf_first_write = otf_csv_path not in _initialized_csvs
            if otf_first_write:
                _initialized_csvs.add(otf_csv_path)
                if os.path.isfile(otf_csv_path):
                    os.remove(otf_csv_path)

        be_fieldnames = ["batch_idx", "graph_idx", "target_binding_E", "computed_binding_E", "binding_E_diff"]

        rows: list[dict] = []

        for atoms, fname, batch_idx, graph_idx, target_E, ref_ads_pos in tqdm(items, desc="Processing", disable=self.opt_graph is not None):
            row = self._process_one(
                atoms=atoms,
                fname=fname,
                batch_idx=batch_idx,
                graph_idx=graph_idx,
                target_E=target_E,
                ref_ads_pos=ref_ads_pos,
            )
            rows.append(row)
            self._all_atoms.append(atoms)

            # OTF: write one CSV row immediately (append mode, one row per graph)
            if otf_csv_path is not None:
                with open(otf_csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=be_fieldnames)
                    if otf_first_write:
                        w.writeheader()
                        otf_first_write = False
                    w.writerow({k: row.get(k, "") for k in be_fieldnames})

        # Disk: write full evaluation CSV at the end
        if self.opt_graph is None:
            out_dir = self.rlxd_POS if self.relax_otf else self.gen_POS
            eval_fieldnames = [
                "batch_idx", "graph_idx", "target_binding_E", "computed_binding_E", "binding_E_diff",
                "vol_ok", "cat_dist_ok", "cat_atom_types_ok", "cat_ads_dist_ok", "ads_dist_ok",
                "is_valid", "is_valid_naive",
            ]
            with open(os.path.join(out_dir, "evaluation.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=eval_fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        return rows

    def _items_from_graph(self):
        prim_batch = self.opt_graph.batch[self.opt_graph.tags == 1]
        ads_batch  = self.opt_graph.batch[self.opt_graph.tags == 2]
        num_graphs = int(self.opt_graph.batch.max().item()) + 1
        items = []
        for b in range(num_graphs):
            atoms       = self._build(b, prim_batch, ads_batch)
            fname       = f"{self.batch_idx}_{b}{self.file_ext}"
            target_E    = self.binding_energies[b]
            ref_ads_pos = self.opt_graph.ref_ads_pos[ads_batch == b]
            items.append((atoms, fname, self.batch_idx, b, target_E, ref_ads_pos))
        return items

    def _items_from_files(self):
        vasp = sorted(glob.glob(os.path.join(self.gen_POS, "*.vasp")))
        files = vasp if vasp else sorted(glob.glob(os.path.join(self.gen_POS, "*.traj")))
        if not files:
            raise FileNotFoundError(f"No .vasp or .traj files found in {self.gen_POS}")

        n = len(files) if self.n_strucs is None else min(self.n_strucs, len(files))
        sample = sorted(random.Random(self.seed).sample(files, n))
        logging.info(f"Processing {len(sample)} structures (seed={self.seed})")

        items = []
        for path in sample:
            key = self._parse_file_key(path=path)
            batch_idx, graph_idx = key if key else (None, None)
            target_E = self.binding_energies.get(key) if key and isinstance(self.binding_energies, dict) else None
            fname = os.path.basename(path)
            try:
                atoms = ase.io.read(path)
            except Exception as e:
                logging.warning(f"Failed to read {path}: {e}")
                atoms = None
            items.append((atoms, fname, batch_idx, graph_idx, target_E, None))
        return items

    def _process_one(
        self,
        atoms: Atoms | None,
        fname: str,
        batch_idx: int | None,
        graph_idx: int | None,
        target_E: float | None,
        ref_ads_pos,
    ) -> dict:
        base_row = {
            "batch_idx":          batch_idx,
            "graph_idx":          graph_idx,
            "target_binding_E":   _fmt(target_E) if target_E is not None else "",
            "computed_binding_E": "",
            "binding_E_diff":     "",
        }

        if not self.sanitize_otf:
            if atoms is not None:
                ase.io.write(os.path.join(self.gen_POS, fname), atoms)
            return base_row

        validity_dict = self.sanitize(
            atoms=atoms,
            catalyst_dist_cutoff=self.catalyst_dist_cutoff,
            # ads_integrity_threshold=self.ads_integrity_threshold,
            # ref_ads_pos=ref_ads_pos,
        )
        atoms_rlxd = self.relax(
            atoms=atoms,
            validity_dict=validity_dict,
            calculator=self.calculator,
            do_relax=self.relax_otf,
        )

        energy_dict = None
        if atoms_rlxd is not None:
            try:
                computed_E = atoms_rlxd.get_potential_energy()
                if computed_E <= self.energy_cutoff:
                    diff = abs(computed_E - target_E) if target_E is not None else None
                    energy_dict = {"computed_binding_E": computed_E, "binding_E_diff": diff}
                else:
                    atoms_rlxd = None
                    validity_dict["is_valid"] = False
            except Exception as e:
                logging.warning(f"Energy extraction failed for {fname}: {e}")
                atoms_rlxd = None
                validity_dict["is_valid"] = False

        if validity_dict["is_valid"] and atoms is not None:
            ase.io.write(os.path.join(self.gen_POS, fname), atoms)
        if atoms_rlxd is not None:
            ase.io.write(os.path.join(self.rlxd_POS, fname), atoms_rlxd)

        return {
            **base_row,
            "computed_binding_E": _fmt(energy_dict["computed_binding_E"]) if energy_dict else "",
            "binding_E_diff":     energy_dict["binding_E_diff"] if energy_dict and energy_dict["binding_E_diff"] is not None else "",
            "vol_ok":             float(validity_dict["vol_ok"]),
            "cat_dist_ok":        float(validity_dict["cat_dist_ok"]),
            "cat_atom_types_ok":  float(validity_dict["cat_atom_types_ok"]),
            "cat_ads_dist_ok":    float(validity_dict["cat_ads_dist_ok"]),
            "ads_dist_ok":        float(validity_dict["ads_dist_ok"]),
            # "ads_integrity_ok":   float(validity_dict["ads_integrity_ok"]) if validity_dict["ads_integrity_ok"] is not None else -1.0,
            "is_valid":           float(validity_dict["is_valid"]),
            "is_valid_naive":     float(validity_dict["is_valid_naive"]),
        }

    @staticmethod
    def _parse_file_key(path: str) -> tuple[int, int] | None:
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    def _build(self, b: int, prim_batch, ads_batch) -> Atoms | None:
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

    @staticmethod
    def sanitize(
        atoms: Atoms | None,
        catalyst_dist_cutoff: float,
        # ads_integrity_threshold: float,
        # ref_ads_pos=None,
    ) -> dict:
        if atoms is None:
            return {
                "vol_ok": False, "cat_dist_ok": False, "cat_atom_types_ok": False,
                "ads_dist_ok": False, "cat_ads_dist_ok": False,
                "is_valid": False, "is_valid_naive": False,
            }

        try:
            vol_ok = float(atoms.get_volume()) >= 0.1
            print(atoms.get_volume())

            tags = atoms.get_tags()
            cat_mask = tags != 2
            ads_mask = tags == 2
            ads_atoms = atoms[ads_mask]
            cat_atoms = atoms[cat_mask]
            ads_idx = np.where(ads_mask)[0]
            cat_idx = np.where(cat_mask)[0]

            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)

            ads_dists = dists[np.ix_(ads_idx, ads_idx)]
            ads_dist_ok = bool(ads_dists.min() >= 0.5)

            # if ref_ads_pos is not None:
            #     ref_np = ref_ads_pos.cpu().float().numpy() if hasattr(ref_ads_pos, "cpu") else np.array(ref_ads_pos, dtype=np.float32)
            #     ref_atoms = Atoms(numbers=ads_atoms.get_atomic_numbers(), positions=ref_np)
            #     ref_ads_dists = ref_atoms.get_all_distances(mic=False)
            #     ads_dists_no_mic = ads_atoms.get_all_distances(mic=False)
            #     ads_integrity_ok = bool(np.abs(ads_dists_no_mic - ref_ads_dists).max() <= ads_integrity_threshold)
            # else:
            #     ads_integrity_ok = False

            cat_dists = dists[np.ix_(cat_idx, cat_idx)]
            cat_atomic_numbers = cat_atoms.get_atomic_numbers()
            atomic_radii = covalent_radii[cat_atomic_numbers]
            sum_atomic_radii = atomic_radii[:, None] + atomic_radii[None, :]
            sum_atomic_radii[sum_atomic_radii == 0.0] = np.inf
            cat_dist_ok = bool((cat_dists >= catalyst_dist_cutoff * sum_atomic_radii).all())
            cat_dist_ok_naive = bool(cat_dists.min() >= 0.5)

            cat_oc20_atom_types = np.arange(22, 84)
            cat_atom_types = np.unique(cat_atomic_numbers)
            cat_atom_types_ok = bool(np.isin(cat_atom_types, cat_oc20_atom_types).any())

            cat_ads_dists = dists[cat_mask, :]
            cat_ads_dist_ok = bool(cat_ads_dists.min() >= 0.5)

        except Exception as e:
            logging.warning(f"batch_relax_w_mlff failed: {e}")
            vol_ok = False
            cat_dist_ok = False
            cat_dist_ok_naive = False
            cat_atom_types_ok = False
            ads_dist_ok = False
            cat_ads_dist_ok = False
            # ads_integrity_ok = False

        is_valid = vol_ok and cat_dist_ok and ads_dist_ok and cat_ads_dist_ok and cat_atom_types_ok

        return {
            "vol_ok":            vol_ok,
            "cat_dist_ok":       cat_dist_ok,
            "cat_atom_types_ok": cat_atom_types_ok,
            "cat_ads_dist_ok":   cat_ads_dist_ok,
            "ads_dist_ok":       ads_dist_ok,
            # "ads_integrity_ok":  ads_integrity_ok,
            "is_valid":          is_valid,
            "is_valid_naive":    cat_dist_ok_naive and ads_dist_ok and cat_ads_dist_ok and vol_ok,
        }

    @staticmethod
    def relax(
        atoms: Atoms | None,
        validity_dict: dict,
        calculator,
        do_relax: bool,
    ) -> Atoms | None:
        if not do_relax or not validity_dict["is_valid"] or atoms is None:
            return None
        return relax_w_mlff(
            atoms=atoms,
            calculator=calculator,
            fmax=0.05,
            steps=200,
            optimizer="BFGS",
        )
