import csv
import logging
import os

import numpy as np
import torch
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii

from adsflow.utils.builder import build_slab_adsorbate, lattice_params_to_matrix
from adsflow.utils.relaxation import _fmt, relax_w_mlff

_initialized_csvs: set = set()


class CatalystForge:
    """Build, filter, relax, and evaluate generated catalyst structures from a single opt_graph.

    Usage:
        cf = CatalystForge(opt_graph, binding_energies, batch_idx, ...)
        metrics_list = cf.evaluate()
    """

    def __init__(
        self,
        opt_graph,
        binding_energies: list[float],
        batch_idx: int,
        calculator,
        run_mlff: bool,
        config_vasp,
        catalyst_dist_cutoff: float,
        gen_POS: str | None,
        rlxd_POS: str | None,
        file_ext: str = ".vasp",
        energy_cutoff: float = 10.0,
    ):
        self.opt_graph = opt_graph
        self.binding_energies = binding_energies
        self.batch_idx = batch_idx
        self.calculator = calculator
        self.run_mlff = run_mlff
        self.config_vasp = config_vasp
        self.catalyst_dist_cutoff = catalyst_dist_cutoff
        self.gen_POS = gen_POS
        self.rlxd_POS = rlxd_POS
        self.file_ext = file_ext
        self.energy_cutoff = energy_cutoff

        self._all_atoms: list[Atoms | None] = []
        self._metrics: list[dict] = []

    def evaluate(self) -> list[dict]:
        prim_batch = self.opt_graph.batch[self.opt_graph.tags == 1]
        ads_batch  = self.opt_graph.batch[self.opt_graph.tags == 2]
        num_graphs = int(self.opt_graph.batch.max().item()) + 1

        os.makedirs(self.gen_POS, exist_ok=True)
        if self.run_mlff:
            os.makedirs(self.rlxd_POS, exist_ok=True)
        csv_dir   = self.rlxd_POS if self.run_mlff else self.gen_POS
        _csv_path = os.path.join(csv_dir, "binding_energies.csv")

        first_write = _csv_path not in _initialized_csvs
        if first_write:
            _initialized_csvs.add(_csv_path)
            if os.path.isfile(_csv_path):
                os.remove(_csv_path)

        fieldnames = ["batch_idx", "graph_idx", "target_binding_E", "computed_binding_E", "binding_E_diff"]

        for b in range(num_graphs):
            target_E = self.binding_energies[b]

            atoms         = self._build(b, prim_batch, ads_batch)
            validity_dict = self._filter(atoms)
            atoms_rlxd    = self._relax(atoms, validity_dict)

            energy_dict = None
            if validity_dict["is_valid"] and self.run_mlff:
                try:
                    energy_dict = {"computed_binding_E": atoms_rlxd.get_potential_energy()}
                except Exception as e:
                    logging.warning(f"energy extraction failed for structure {b}: {e}")
                    atoms_rlxd = None
                    validity_dict["is_valid"] = False

            if energy_dict is not None and energy_dict["computed_binding_E"] > self.energy_cutoff:
                energy_dict = None
                atoms_rlxd = None
                validity_dict["is_valid"] = False

            if energy_dict is not None:
                energy_dict["binding_E_diff"] = abs(energy_dict["computed_binding_E"] - target_E)
            row = {
                "batch_idx": self.batch_idx,
                "graph_idx": b,
                "target_binding_E": _fmt(target_E),
                "computed_binding_E": _fmt(energy_dict["computed_binding_E"]) if energy_dict else "",
                "binding_E_diff": _fmt(energy_dict["binding_E_diff"]) if energy_dict else "",
            }
            with open(_csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if first_write:
                    w.writeheader()
                    first_write = False
                w.writerow(row)

            fname = f"{self.batch_idx}_{b}{self.file_ext}"
            if validity_dict["is_valid"] and atoms is not None:
                ase.io.write(os.path.join(self.gen_POS, fname), atoms)
            if atoms_rlxd is not None:
                ase.io.write(os.path.join(self.rlxd_POS, fname), atoms_rlxd)

            metrics = {
                "vol_ok":                float(validity_dict["vol_ok"]),
                "cat_dist_ok":           float(validity_dict["cat_dist_ok"]),
                "cat_atom_types_ok":     float(validity_dict["cat_atom_types_ok"]),
                "cat_ads_dist_ok":       float(validity_dict["cat_ads_dist_ok"]),
                "ads_dist_ok":           float(validity_dict["ads_dist_ok"]),
                "sample_is_valid":       float(validity_dict["is_valid"]),
                "sample_is_valid_naive": float(validity_dict["is_valid_naive"]),
            }
            if energy_dict is not None and target_E is not None:
                metrics["binding_E_diff"] = energy_dict["binding_E_diff"]
            self._metrics.append(metrics)
            self._all_atoms.append(atoms)

        return self._metrics

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

    def _filter(self, atoms: Atoms | None) -> dict:
        if atoms is None:
            return {
                "vol_ok": False, "cat_dist_ok": False, "cat_atom_types_ok": False,
                "ads_dist_ok": False, "cat_ads_dist_ok": False,
                "is_valid": False, "is_valid_naive": False,
            }

        try:
            vol_ok = float(atoms.get_volume()) >= 0.1
            tags = atoms.get_tags()
            cat_mask = tags != 2
            ads_mask = tags == 2

            cat_atoms = atoms[cat_mask]
            dists = atoms.get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)

            ads_idx = np.where(ads_mask)[0]
            ads_dists = dists[np.ix_(ads_idx, ads_idx)]
            ads_dist_ok = bool(ads_dists.min() >= 0.5)

            cat_idx = np.where(cat_mask)[0]
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

        except Exception:
            cat_dist_ok = False
            cat_dist_ok_naive = False
            cat_atom_types_ok = False
            vol_ok = False
            ads_dist_ok = False
            cat_ads_dist_ok = False

        return {
            "vol_ok":            vol_ok,
            "cat_dist_ok":       cat_dist_ok,
            "cat_atom_types_ok": cat_atom_types_ok,
            "cat_ads_dist_ok":   cat_ads_dist_ok,
            "ads_dist_ok":       ads_dist_ok,
            "is_valid":          vol_ok and cat_dist_ok and ads_dist_ok and cat_ads_dist_ok and cat_atom_types_ok,
            "is_valid_naive":    cat_dist_ok_naive and ads_dist_ok and cat_ads_dist_ok and vol_ok,
        }

    def _relax(self, atoms: Atoms | None, validity_dict: dict) -> Atoms | None:
        if not validity_dict["is_valid"] or not self.run_mlff:
            return None
        return relax_w_mlff(
            atoms=atoms,
            calculator=self.calculator,
            fmax=0.05,
            steps=200,
            optimizer="BFGS",
        )
