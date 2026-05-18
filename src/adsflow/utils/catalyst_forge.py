import csv
import logging
import os

import ase.io
from ase import Atoms

from adsflow.utils.builder import forge_catalyst
from adsflow.utils.evaluator import check_validity
from adsflow.utils.relaxation import _fmt, batch_relax_w_mlff, relax_struc

_initialized_csvs: set = set()


class CatalystForge:
    """Build, relax, and evaluate generated catalyst structures from a single opt_graph.

    Usage:
        cf = CatalystForge(opt_graph, binding_energies, batch_idx, ...)
        cf.build()
        cf.relax()
        metrics_list = cf.evaluate()
    """

    def __init__(
        self,
        opt_graph,
        binding_energies: list[float],
        batch_idx: int,
        calculator,
        run_mlff: bool,
        batched_relaxation: bool,
        config_vasp,
        catalyst_dist_cutoff: float,
        gen_POS: str | None,
        rlxd_POS: str | None,
        file_ext: str = ".vasp",
    ):
        self.opt_graph = opt_graph
        self.binding_energies = binding_energies
        self.batch_idx = batch_idx
        self.calculator = calculator
        self.run_mlff = run_mlff
        self.batched_relaxation = batched_relaxation
        self.config_vasp = config_vasp
        self.catalyst_dist_cutoff = catalyst_dist_cutoff
        self.gen_POS = gen_POS
        self.rlxd_POS = rlxd_POS
        self.file_ext = file_ext

        self._all_atoms: list[Atoms | None] = []
        self._relaxed_atoms: list[Atoms | None] = []
        self._validity_results: list[dict] = []
        self._metrics: list[dict] = []

    def build(self) -> list[Atoms | None]:
        prim_batch = self.opt_graph.batch[self.opt_graph.tags == 1]
        ads_batch  = self.opt_graph.batch[self.opt_graph.tags == 2]
        num_graphs = int(self.opt_graph.batch.max().item()) + 1
        for b in range(num_graphs):
            atoms = forge_catalyst(self.opt_graph, b, prim_batch, ads_batch)
            self._all_atoms.append(atoms)
        return self._all_atoms

    def filter(self) -> list[dict]:
        for i, atoms in enumerate(self._all_atoms):
            result = check_validity(atoms, self.catalyst_dist_cutoff)
            self._validity_results.append(result)
            if not result["is_valid"]:
                self._all_atoms[i] = None
        return self._validity_results

    def relax(self) -> list[Atoms | None]:
        self._relaxed_atoms = [None] * len(self._all_atoms)
        if not self.run_mlff:
            return self._relaxed_atoms

        valid_indices = [i for i, a in enumerate(self._all_atoms) if a is not None]
        if not valid_indices:
            return self._relaxed_atoms

        if self.batched_relaxation:
            relaxed_list = batch_relax_w_mlff(
                [self._all_atoms[i] for i in valid_indices], self.calculator
            )
            for i, rlxd in zip(valid_indices, relaxed_list):
                self._relaxed_atoms[i] = rlxd
        else:
            for i in valid_indices:
                atoms_rlxd, _ = relax_struc(
                    atoms=self._all_atoms[i],
                    calculator=self.calculator,
                    config_vasp=self.config_vasp,
                )
                self._relaxed_atoms[i] = atoms_rlxd

        return self._relaxed_atoms

    def evaluate(self) -> list[dict]:
        for i, (atoms, atoms_rlxd, target_E, result) in enumerate(
            zip(self._all_atoms, self._relaxed_atoms, self.binding_energies, self._validity_results)
        ):
            energy_dict = None

            if result["is_valid"] and atoms_rlxd is not None:
                out_dir = os.path.join(self.rlxd_POS, f"{self.batch_idx}_{i}")
                _csv_path = os.path.join(out_dir, "binding_energies.csv")
                try:
                    E = atoms_rlxd.get_potential_energy()
                    energy_dict = {"computed_binding_E": E}
                except Exception as e:
                    logging.warning(f"energy extraction failed for structure {i}: {e}")
                    atoms_rlxd = None
                    result["is_valid"] = False
            else:
                out_dir = os.path.join(self.gen_POS, f"{self.batch_idx}_{i}")
                _csv_path = os.path.join(out_dir, "binding_energies.csv")

            if energy_dict is not None and energy_dict["computed_binding_E"] > 10:
                energy_dict = None
                atoms_rlxd = None
                result["is_valid"] = False

            os.makedirs(out_dir, exist_ok=True)
            first_write = _csv_path not in _initialized_csvs
            if first_write:
                _initialized_csvs.add(_csv_path)
                if os.path.isfile(_csv_path):
                    os.remove(_csv_path)
            if energy_dict is not None:
                energy_dict["binding_E_diff"] = abs(energy_dict["computed_binding_E"] - target_E)
                fieldnames = ["batch_idx", "graph_idx", "target_binding_E", "computed_binding_E", "binding_E_diff"]
                row = {
                    "batch_idx": self.batch_idx,
                    "graph_idx": i,
                    "target_binding_E": _fmt(target_E),
                    "computed_binding_E": _fmt(energy_dict["computed_binding_E"]),
                    "binding_E_diff": _fmt(energy_dict["binding_E_diff"]),
                }
            else:
                fieldnames = ["batch_idx", "graph_idx", "target_binding_E"]
                row = {
                    "batch_idx": self.batch_idx,
                    "graph_idx": i,
                    "target_binding_E": _fmt(target_E),
                }
            with open(_csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if first_write:
                    w.writeheader()
                w.writerow(row)

            fname = f"{self.batch_idx}_{i}{self.file_ext}"
            if result["is_valid"]:
                gen_dir = os.path.join(self.gen_POS, f"{self.batch_idx}_{i}")
                os.makedirs(gen_dir, exist_ok=True)
                ase.io.write(os.path.join(gen_dir, fname), atoms)
                if atoms_rlxd is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    ase.io.write(os.path.join(out_dir, fname), atoms_rlxd)

            metrics = {
                "vol_ok":                float(result["vol_ok"]),
                "cat_dist_ok":           float(result["cat_dist_ok"]),
                "cat_atom_types_ok":     float(result["cat_atom_types_ok"]),
                "cat_ads_dist_ok":       float(result["cat_ads_dist_ok"]),
                "ads_dist_ok":           float(result["ads_dist_ok"]),
                "sample_is_valid":       float(result["is_valid"]),
                "sample_is_valid_naive": float(result["is_valid_naive"]),
            }
            if energy_dict is not None and target_E is not None:
                metrics["binding_E_diff"] = energy_dict["binding_E_diff"]

            self._metrics.append(metrics)

        return self._metrics
