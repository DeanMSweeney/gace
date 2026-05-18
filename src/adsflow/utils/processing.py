import csv
import logging
import os

import ase.io
import numpy as np
from ase import Atoms

from adsflow.utils.evaluator import check_validity
from adsflow.utils.relaxation import _fmt, relax_struc

_initialized_csvs: set = set()


def process_sample(
    atoms,
    graph_idx: int,
    batch_idx: int,
    *,
    calculator,
    run_mlff: bool,
    config_vasp,
    catalyst_dist_cutoff: float,
    gen_POS: str | None,
    rlxd_POS: str | None,
    target_binding_E: float | None,
    file_ext: str = ".vasp",
    atoms_rlxd: Atoms | None = None,
    batched_relaxation: bool = False,
) -> tuple:
    """Post-process a single generated sample.

    Relaxes the structure, checks validity, writes output files, and returns
    the relaxed atoms and a metrics dict.

    Args:
        atoms: Raw ASE Atoms object from forge_catalyst.
        graph_idx: Index of this graph within the batch (used for file naming).
        batch_idx: Index of the batch (used for file naming and CSV).
        calculator: MLFF calculator instance (or None).
        run_mlff: Whether to run structure relaxation.
        config_vasp: VASP config dict (or None).
        catalyst_dist_cutoff: Scale factor for covalent radii in validity check.
        gen_POS: Directory to write unrelaxed structures (or None).
        rlxd_POS: Directory to write relaxed structures (or None).
        target_binding_E: Target binding energy (or None).
        file_ext: File extension for written structures (".vasp" or ".traj").
        atoms_rlxd: Pre-relaxed ASE Atoms; when provided, internal relaxation is skipped.

    Returns:
        (atoms_rlxd, metrics) where metrics is a dict of validity floats.
    """


    result = check_validity(atoms, catalyst_dist_cutoff)

    if result["is_valid"] and run_mlff:
        rlxd_POS = os.path.join(rlxd_POS, f"{batch_idx}_{graph_idx}")
        _csv_path = os.path.join(rlxd_POS, "binding_energies.csv")
        if batched_relaxation:
            try:
                E_slab_ads = atoms_rlxd.get_potential_energy()
                energy_dict = {"computed_binding_E": E_slab_ads}
            except Exception as e:
                logging.warning(f"energy extraction from pre-relaxed atoms failed: {e}")
                atoms_rlxd = None
                energy_dict = None
                result["is_valid"] = False
        else:
            atoms_rlxd, energy_dict = relax_struc(
                atoms=atoms,
                calculator=calculator,
                config_vasp=config_vasp,
            )
            if atoms_rlxd is None and energy_dict is None:
                result["is_valid"] = False
    else:
        gen_POS = os.path.join(gen_POS, f"{batch_idx}_{graph_idx}")
        _csv_path = os.path.join(gen_POS, "binding_energies.csv")
        atoms_rlxd = None
        energy_dict = None

    if energy_dict is not None and energy_dict["computed_binding_E"] > 10: # discard very unstable structures
        energy_dict = None 
        atoms_rlxd = None 
        result["is_valid"] = False

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
                "computed_binding_E", "binding_E_diff",
            ]
            energy_dict["binding_E_diff"] = abs(energy_dict["computed_binding_E"] - target_binding_E)
            row = {
                "batch_idx": batch_idx,
                "graph_idx": graph_idx,
                "target_binding_E": _fmt(target_binding_E),
                "computed_binding_E": _fmt(energy_dict["computed_binding_E"]),
                "binding_E_diff": _fmt(energy_dict["binding_E_diff"]),
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
    
    metrics = {
        "vol_ok":            float(result["vol_ok"]),
        "cat_dist_ok":       float(result["cat_dist_ok"]),
        "cat_atom_types_ok": float(result["cat_atom_types_ok"]),
        "cat_ads_dist_ok":   float(result["cat_ads_dist_ok"]),
        "ads_dist_ok":       float(result["ads_dist_ok"]),
        "sample_is_valid":   float(result["is_valid"]),
        "sample_is_valid_naive":    float(result["is_valid_naive"]),
    }

        
    if energy_dict is not None and target_binding_E is not None:
        metrics["binding_E_diff"] = energy_dict["binding_E_diff"]

    fname = f"{batch_idx}_{graph_idx}{file_ext}"

    if result["is_valid"]:
        os.makedirs(gen_POS, exist_ok=True)
        ase.io.write(os.path.join(gen_POS, fname), atoms)

        if run_mlff and atoms_rlxd is not None:
            os.makedirs(rlxd_POS, exist_ok=True)
            ase.io.write(os.path.join(rlxd_POS, fname), atoms_rlxd)

        
    return atoms_rlxd, metrics
