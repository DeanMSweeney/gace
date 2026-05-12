import csv
import os

import ase.io
import numpy as np

from adsflow.utils.evaluator import check_validity
from adsflow.utils.relaxation import _fmt, relax_struc

_initialized_csvs: set = set()


def process_sample(
    atoms,
    target_ads_dists: np.ndarray,
    graph_idx: int,
    batch_idx: int,
    *,
    calculator,
    run_mlff: bool,
    config_vasp,
    calc_binding_energy: bool,
    covalent_radii_scale: float,
    gen_POS: str | None,
    rlxd_POS: str | None,
    target_binding_E: float | None,
    file_ext: str = ".vasp",
) -> tuple:
    """Post-process a single generated sample.

    Relaxes the structure, checks validity, writes output files, and returns
    the relaxed atoms and a metrics dict.

    Args:
        atoms: Raw ASE Atoms object from forge_catalyst.
        target_ads_dists: (N_ads, 15) pairwise distance array for this graph.
        graph_idx: Index of this graph within the batch (used for file naming).
        batch_idx: Index of the batch (used for file naming and CSV).
        calculator: MLFF calculator instance (or None).
        run_mlff: Whether to run structure relaxation.
        config_vasp: VASP config dict (or None).
        calc_binding_energy: Whether relax_struc should compute binding energy.
        covalent_radii_scale: Scale factor for covalent radii in validity check.
        gen_POS: Directory to write unrelaxed structures (or None).
        rlxd_POS: Directory to write relaxed structures (or None).
        target_binding_E: Target binding energy (or None).
        file_ext: File extension for written structures (".vasp" or ".traj").

    Returns:
        (atoms_rlxd, metrics) where metrics is a dict of validity floats.
    """
    _csv_path = (
        os.path.join(rlxd_POS, "binding_energies.csv")
        if calc_binding_energy and rlxd_POS
        else (os.path.join(gen_POS, "binding_energies.csv") if gen_POS else None)
    )

    atoms_rlxd, energy_dict = relax_struc(
        atoms=atoms,
        calculator=calculator,
        run_mlff=run_mlff,
        config_vasp=config_vasp,
        calc_binding_energy=calc_binding_energy,
    )

    if _csv_path is not None:
        os.makedirs(os.path.dirname(_csv_path), exist_ok=True)
        first_write = _csv_path not in _initialized_csvs
        if first_write:
            _initialized_csvs.add(_csv_path)
            if os.path.isfile(_csv_path):
                os.remove(_csv_path)
        if energy_dict is not None:
            fieldnames = [
                "batch_idx", "graph_idx", "target_binding_E",
                "computed_binding_E", "ERROR",
            ]
            energy_dict["ERROR"] = abs(energy_dict["computed_binding_E"] - target_binding_E)
            row = {
                "batch_idx": batch_idx,
                "graph_idx": graph_idx,
                "target_binding_E": _fmt(target_binding_E),
                "computed_binding_E": _fmt(energy_dict["computed_binding_E"]),
                "ERROR": _fmt(energy_dict["ERROR"]),
            }
        else:
            fieldnames = ["batch_idx", "graph_idx", "target_binding_E"]
            row = {
                "batch_idx": batch_idx,
                "graph_idx": graph_idx,
                "target_binding_E": _fmt(target_binding_E),
            }
        
        with open(_csv_path, "a", newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=fieldnames)
            if first_write:
                _w.writeheader()
            _w.writerow(row)
        

    atoms_rlxd = atoms_rlxd if run_mlff else atoms
    result = check_validity(atoms_rlxd, covalent_radii_scale, target_ads_dists=target_ads_dists)

    metrics = {
        "vol_ok":            float(result["vol_ok"]),
        "cat_dist_ok_naive": float(result["cat_dist_ok_naive"]),
        "cat_dist_ok":       float(result["cat_dist_ok"]),
        "ads_dist_ok":       float(result["ads_dist_ok"]),
        "sample_is_valid":   float(result["is_valid"]),
    }
    if result["ads_dist_mae"] is not None:
        metrics["ads_dist_mae"] = float(result["ads_dist_mae"])
    if energy_dict is not None and target_binding_E is not None:
        metrics["binding_E_mae"] = energy_dict["ERROR"]

    fname = f"{batch_idx}_{graph_idx}{file_ext}"
    if gen_POS and result["is_valid"]:
        os.makedirs(gen_POS, exist_ok=True)
        ase.io.write(os.path.join(gen_POS, fname), atoms)

    if rlxd_POS and run_mlff and result["is_valid"]:
        os.makedirs(rlxd_POS, exist_ok=True)
        ase.io.write(os.path.join(rlxd_POS, fname), atoms_rlxd)

    return atoms_rlxd, metrics
