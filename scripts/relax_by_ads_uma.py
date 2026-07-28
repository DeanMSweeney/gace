"""Relax .traj structures with UMA, filtered by adsorbate id, and save adsorption energies to CSV.

Same selection interface as relax_by_ads.py — the filename->ads_id CSV from
list_traj_by_ads.py picks which structures to relax — but the calculator is UMA
(fairchem v2) instead of an OCPCalculator, and no gace.utils import is needed.

Adsorption energy is computed the OC20 way, so it is directly comparable to the
target_binding_E stored in atoms.info:

    E_ads = E(relaxed slab+adsorbate) - E(relaxed bare slab) - sum(gas-phase refs)

Outputs land in <out_dir>/ads_<id>/:
    energies.csv  paired ads_<id>_adsorption_E / ads_<id>_target_E columns, one row per structure
    details.csv   one row per structure with both relaxation energies, step counts and errors

Usage:
    python scripts/relax_by_ads_uma.py \
        --traj_dir for_uma/1_threshold \
        --ads_map traj_by_ads.csv \
        --ads_ids 5
"""

import argparse
import contextlib
import csv
import functools
import io
import logging
import multiprocessing
import statistics
import sys
from pathlib import Path

import pandas as pd
import torch
from ase.constraints import FixAtoms
from ase.io import read
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# OC20 reference energies computed via UMA from gas-phase molecules
# C = E(CO) - E(H2O) + E(H2), H = 0.5*E(H2), N = 0.5*E(N2), O = E(H2O) - E(H2)
OC20_GAS_PHASE_ENERGIES = {
    "H": -3.48483361833793,
    "O": -7.185616160375758,
    "C": -7.232295041080779,
    "N": -8.09079187764214,
}

# Written to details.csv; the wide energies.csv is built from a subset of these
DETAIL_FIELDS = [
    "filename",
    "ads_id",
    "adsorption_E",
    "target_binding_E",
    "binding_E_diff",
    "e_sys_relaxed",
    "e_slab_relaxed",
    "e_adsorbate_ref",
    "converged_system",
    "converged_slab",
    "steps_system",
    "steps_slab",
    "error",
]


def get_adsorbate_ref_energy(atoms) -> float:
    """Sum of gas-phase reference energies over the adsorbate atoms (tag == 2)."""
    total = 0.0
    for symbol, tag in zip(atoms.get_chemical_symbols(), atoms.get_tags()):
        if tag == 2:
            if symbol not in OC20_GAS_PHASE_ENERGIES:
                raise ValueError(f"No UMA reference energy for adsorbate element '{symbol}'")
            total += OC20_GAS_PHASE_ENERGIES[symbol]
    return total


def make_optimizer(name: str):
    from ase.optimize import BFGS, FIRE, LBFGS

    return {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}[name]


# ---------------------------------------------------------------------------
# Per-structure relaxation (runs inside a worker thread of a GPU process)
# ---------------------------------------------------------------------------

def process_single_traj(
    task: tuple,
    batch_predict_unit=None,
    uma_task: str = "oc20",
    fmax: float = 0.05,
    steps: int = 200,
    optimizer: str = "LBFGS",
    energy_cutoff: float = 10.0,
    relaxed_dir: str | None = None,
) -> dict:
    from fairchem.core import FAIRChemCalculator

    traj_path, ads_id = task
    traj_path = Path(traj_path)

    result = {field: None for field in DETAIL_FIELDS}
    result.update({
        "filename": traj_path.name,
        "ads_id": ads_id,
        "converged_system": False,
        "converged_slab": False,
        "steps_system": 0,
        "steps_slab": 0,
    })

    try:
        atoms = read(traj_path)
        result["target_binding_E"] = atoms.info.get("target_binding_E")

        if 2 not in atoms.get_tags():
            result["error"] = "No adsorbate atoms (tags == 2)"
            return result

        # Full slab+adsorbate and the bare slab, both with subsurface atoms fixed
        system = atoms.copy()
        if 0 in system.get_tags():
            system.set_constraint(FixAtoms(indices=[a.index for a in system if a.tag == 0]))

        slab = atoms.copy()[atoms.get_tags() != 2]
        if 0 in slab.get_tags():
            slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 0]))

        e_adsorbate_ref = get_adsorbate_ref_energy(atoms=atoms)
        result["e_adsorbate_ref"] = e_adsorbate_ref

        calculator = FAIRChemCalculator(batch_predict_unit, task_name=uma_task)
        opt_class = make_optimizer(name=optimizer)

        system.calc = calculator
        opt_sys = opt_class(system, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_system = opt_sys.run(fmax=fmax, steps=steps)
        e_sys_relaxed = system.get_potential_energy()

        slab.calc = calculator
        opt_slab = opt_class(slab, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_slab = opt_slab.run(fmax=fmax, steps=steps)
        e_slab_relaxed = slab.get_potential_energy()

        result.update({
            "e_sys_relaxed": e_sys_relaxed,
            "e_slab_relaxed": e_slab_relaxed,
            "converged_system": bool(converged_system),
            "converged_slab": bool(converged_slab),
            "steps_system": opt_sys.get_number_of_steps(),
            "steps_slab": opt_slab.get_number_of_steps(),
        })

        e_ads = e_sys_relaxed - e_slab_relaxed - e_adsorbate_ref
        if abs(e_ads) > energy_cutoff:
            result["error"] = f"|E_ads|={abs(e_ads):.3f} eV exceeds cutoff {energy_cutoff}"
            return result

        result["adsorption_E"] = e_ads
        target_E = result["target_binding_E"]
        if target_E is not None:
            result["binding_E_diff"] = abs(e_ads - target_E)

        if relaxed_dir is not None:
            out_path = Path(relaxed_dir) / traj_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            system.calc = None
            system.info["adsorption_E"] = e_ads
            if result["binding_E_diff"] is not None:
                system.info["binding_E_diff"] = result["binding_E_diff"]
            system.write(str(out_path))

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# GPU worker process
# ---------------------------------------------------------------------------

def run_gpu_process(gpu_id: int, tasks: list, num_workers: int, worker_kwargs: dict) -> list:
    """Load UMA once per process, then relax this process's share of the structures."""
    import os

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate import InferenceBatcher

    try:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = "cuda"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = "cpu"

        predict_unit = pretrained_mlip.get_predict_unit(worker_kwargs.pop("uma_model"), device=device)
        batcher = InferenceBatcher(
            predict_unit,
            concurrency_backend_options={"max_workers": num_workers},
        )
        process_fn = functools.partial(
            process_single_traj,
            batch_predict_unit=batcher.batch_predict_unit,
            **worker_kwargs,
        )
        return list(batcher.executor.map(process_fn, tasks))

    except Exception:
        import traceback

        print(traceback.format_exc(), file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Selection + reporting
# ---------------------------------------------------------------------------

def load_ads_map(ads_map: Path, keep_ads_ids: set[int]) -> dict[str, int]:
    """filename -> ads_id for files whose id parses as an int and is requested."""
    ads_id_by_file = {}
    with open(ads_map, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ads_id = int(row["ads_id"])
            except ValueError:
                continue  # no_sid / unreadable / sid_not_in_mapping labels
            if ads_id in keep_ads_ids:
                ads_id_by_file[row["filename"]] = ads_id
    return ads_id_by_file


def log_summary(rows: list[dict], ads_id: int) -> None:
    diffs = [r["binding_E_diff"] for r in rows if r.get("binding_E_diff") is not None]
    if not diffs:
        log.info(f"ads_id={ads_id}: no structures with both an adsorption energy and a target")
        return
    log.info(
        f"ads_id={ads_id}: n={len(diffs)} "
        f"mean|dE|={statistics.mean(diffs):.4f} "
        f"median|dE|={statistics.median(diffs):.4f} "
        f"min={min(diffs):.4f} max={max(diffs):.4f} eV"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, required=True, help="Directory containing .traj files")
    parser.add_argument("--ads_map", type=Path, required=True, help="filename,ads_id CSV from list_traj_by_ads.py")
    parser.add_argument("--ads_ids", type=int, nargs="+", required=True, help="Adsorbate ids to keep")
    parser.add_argument("--out_dir", type=Path, default=Path("."),
                        help="Root output directory; CSVs go in <out_dir>/ads_<id>/")
    parser.add_argument("--uma_model", type=str, default="uma-s-1p1", help="UMA checkpoint name")
    parser.add_argument("--uma_task", type=str, default="oc20", help="UMA task head")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="LBFGS", choices=["BFGS", "LBFGS", "FIRE"])
    parser.add_argument("--energy_cutoff", type=float, default=10.0,
                        help="Reject |E_ads| above this (eV), matching the CatFlow convention")
    parser.add_argument("--num_workers", type=int, default=16, help="Concurrent relaxations per GPU process")
    parser.add_argument("--num_gpus", type=int, default=None, help="GPUs to use (default: all available)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPUs are visible")
    parser.add_argument("--save_relaxed", action="store_true",
                        help="Write relaxed structures to <out_dir>/ads_<id>/relaxed/")
    parser.add_argument("--keep_unconverged", action="store_true",
                        help="Include unconverged relaxations in energies.csv (they are always in details.csv)")
    args = parser.parse_args()

    keep_ads_ids = sorted(set(args.ads_ids))

    # One folder per run, named by the requested adsorbate id(s)
    run_dir = args.out_dir / ("ads_" + "_".join(str(i) for i in keep_ads_ids))
    run_dir.mkdir(parents=True, exist_ok=True)
    relaxed_dir = str(run_dir / "relaxed") if args.save_relaxed else None

    ads_id_by_file = load_ads_map(ads_map=args.ads_map, keep_ads_ids=set(keep_ads_ids))
    log.info(f"{len(ads_id_by_file)} files in {args.ads_map} match the requested adsorbate ids")

    tasks, missing = [], 0
    for filename, ads_id in sorted(ads_id_by_file.items()):
        traj_path = args.traj_dir / filename
        if not traj_path.is_file():
            missing += 1
            continue
        tasks.append((str(traj_path), ads_id))
    if missing:
        log.warning(f"{missing} listed files are not present under {args.traj_dir}, skipping")
    if not tasks:
        raise FileNotFoundError(f"No matching .traj files found under {args.traj_dir}")

    worker_kwargs = {
        "uma_model": args.uma_model,
        "uma_task": args.uma_task,
        "fmax": args.fmax,
        "steps": args.steps,
        "optimizer": args.optimizer,
        "energy_cutoff": args.energy_cutoff,
        "relaxed_dir": relaxed_dir,
    }

    num_gpus = 0
    if not args.cpu and torch.cuda.is_available():
        num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()

    all_results: list[dict] = []
    if num_gpus > 0:
        log.info(f"Relaxing {len(tasks)} structures on {num_gpus} GPU(s), {args.num_workers} concurrent each")
        tasks_per_gpu = [[] for _ in range(num_gpus)]
        for i, task in enumerate(tasks):
            tasks_per_gpu[i % num_gpus].append(task)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_gpus) as pool:
            async_results = [
                pool.apply_async(run_gpu_process, (gpu_id, chunk, args.num_workers, dict(worker_kwargs)))
                for gpu_id, chunk in enumerate(tasks_per_gpu)
                if chunk
            ]
            for res in tqdm(async_results, desc="GPU batches"):
                all_results.extend(res.get())
    else:
        log.info(f"Relaxing {len(tasks)} structures on CPU, {args.num_workers} concurrent")
        all_results = run_gpu_process(-1, tasks, args.num_workers, dict(worker_kwargs))

    all_results.sort(key=lambda r: (r["ads_id"], r["filename"]))

    details_path = run_dir / "details.csv"
    with open(details_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k) for k in DETAIL_FIELDS})
    log.info(f"Wrote {details_path}: {len(all_results)} rows")

    n_errors = sum(1 for r in all_results if r.get("error"))
    n_unconverged = sum(1 for r in all_results if not (r.get("converged_system") and r.get("converged_slab")))
    log.info(f"Errors: {n_errors}/{len(all_results)}  Unconverged: {n_unconverged}/{len(all_results)}")

    # One CSV with paired columns per adsorbate id; ragged columns padded with NaN
    columns = {}
    for ads_id in keep_ads_ids:
        rows = [
            r for r in all_results
            if r["ads_id"] == ads_id
            and r.get("adsorption_E") is not None
            and (args.keep_unconverged or (r["converged_system"] and r["converged_slab"]))
        ]
        log_summary(rows=rows, ads_id=ads_id)
        columns[f"ads_{ads_id}_adsorption_E"] = [r["adsorption_E"] for r in rows]
        # Appended in lockstep with energies so rows align across the paired columns
        columns[f"ads_{ads_id}_target_E"] = [
            r["target_binding_E"] if r["target_binding_E"] is not None else float("nan") for r in rows
        ]

    df = pd.DataFrame.from_dict(columns, orient="index").transpose()
    energies_path = run_dir / "energies.csv"
    df.to_csv(energies_path, index=False)
    log.info(f"Wrote {energies_path}: " + ", ".join(f"{c}={df[c].notna().sum()}" for c in df.columns))


if __name__ == "__main__":
    main()
