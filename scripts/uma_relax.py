"""Evaluate pre-sanitized .traj structures using UMA relaxation and compute adsorption energies.

Replicates the CatFlow E_ads_eval_batch workflow applied to structures that have already
been sanitized and have atoms.info["target_binding_E"] set.

Adsorption energy formula:
    ΔE_ads = E_sys_relaxed - E_slab_relaxed - E_ads_ref

where E_ads_ref sums OC20 UMA reference energies over adsorbate atoms (tags == 2).

Outputs:
  <relaxed_strucs>/relaxed_strucs_data.csv   — per-structure metrics
  <relaxed_strucs>/relaxed/                  — relaxed .traj files grouped by threshold subdir
  <output>                                   — summary report

Arguments
---------
--strucs_dir       Root directory to search recursively for .traj files.
--relaxed_strucs   Output directory. Default: ./relaxed_strucs
--uma_model        UMA model name. Default: uma-s-1p1
--fmax             Force convergence criterion (eV/Å). Default: 0.05
--steps            Max LBFGS steps. Default: 500
--energy_cutoff    Max |ΔE_ads| (eV) before structure is discarded. Default: 10.0
--n_strucs         Number of structures to randomly sample (default: all).
--seed             Random seed for sampling. Default: 42
--num_workers      Concurrent relaxations per GPU/CPU process. Default: 16
--num_gpus         Number of GPUs to use. Default: all available (0 = CPU).
--cpu              Force CPU execution.
--output           Summary report path. Default: ./eval_traj_report.txt

Example
-------
    python scripts/uma_relax.py \\
        --strucs_dir ./sanitized_strucs \\
        --relaxed_strucs ./relaxed_strucs \\
        --output ./uma_report.txt
"""

import argparse
import contextlib
import csv
import functools
import io
import json
import logging
import multiprocessing
import os
import pathlib
import random
import statistics
import sys
import warnings
from typing import Dict

import numpy as np
import torch
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import LBFGS
from tqdm import tqdm

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# OC20 reference energies computed via UMA from gas-phase molecules
# C = E(CO) - E(H2O) + E(H2), H = 0.5*E(H2), N = 0.5*E(N2), O = E(H2O) - E(H2)
OC20_GAS_PHASE_ENERGIES = {
    "H": -3.48483361833793,
    "O": -7.185616160375758,
    "C": -7.232295041080779,
    "N": -8.09079187764214,
}

_BINDING_E_THRESHOLD = 0.10
_THRESHOLD_MULTIPLIERS = [1, 2, 3, 5, 10]
_CSV_DECIMALS = 6
_FMT = f".{_CSV_DECIMALS}f"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_fmt(v) -> str:
    if isinstance(v, float):
        return format(v, _FMT)
    return str(v) if v is not None else ""


def _threshold_flags(diff: float | None) -> dict:
    if diff is None:
        return {f"within_{m}T": None for m in _THRESHOLD_MULTIPLIERS}
    return {f"within_{m}T": float(diff <= m * _BINDING_E_THRESHOLD) for m in _THRESHOLD_MULTIPLIERS}


def _threshold_subdir(diff: float | None) -> str:
    if diff is not None:
        for m in _THRESHOLD_MULTIPLIERS:
            if diff <= m * _BINDING_E_THRESHOLD:
                return f"{m}_threshold"
    return "inf_threshold"


def get_adsorbate_ref_energy(atoms) -> float:
    symbols = atoms.get_chemical_symbols()
    tags = atoms.get_tags()
    total = 0.0
    for sym, tag in zip(symbols, tags):
        if tag == 2:
            if sym not in OC20_GAS_PHASE_ENERGIES:
                raise ValueError(f"No UMA reference energy for adsorbate element '{sym}'")
            total += OC20_GAS_PHASE_ENERGIES[sym]
    return total


# ---------------------------------------------------------------------------
# Per-structure relaxation (runs inside worker process)
# ---------------------------------------------------------------------------

def process_single_traj(args, batch_predict_unit=None) -> Dict:
    from fairchem.core import FAIRChemCalculator

    traj_path, target_E, output_dir = args
    fname = pathlib.Path(traj_path).name

    result = {
        "fname":              fname,
        "target_binding_E":  target_E,
        "computed_binding_E": None,
        "binding_E_diff":    None,
        "e_sys_relaxed":     None,
        "e_slab_relaxed":    None,
        "e_adsorbate":       None,
        "converged_system":  False,
        "converged_slab":    False,
        "steps_system":      0,
        "steps_slab":        0,
        **{f"within_{m}T": None for m in _THRESHOLD_MULTIPLIERS},
        "error":             None,
    }

    try:
        atoms = read(traj_path)

        # Override target_E from atoms.info if not provided externally
        if target_E is None:
            target_E = atoms.info.get("target_binding_E")
            result["target_binding_E"] = target_E

        # Build system (full slab+adsorbate) and bare slab
        system = atoms.copy()
        if 0 in system.get_tags():
            system.set_constraint(FixAtoms(indices=[a.index for a in system if a.tag == 0]))

        slab = atoms.copy()[atoms.get_tags() != 2]
        if 0 in slab.get_tags():
            slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 0]))

        adsorbate = atoms[atoms.get_tags() == 2]
        if len(adsorbate) == 0:
            result["error"] = "No adsorbate atoms (tags == 2)"
            return result

        e_adsorbate = get_adsorbate_ref_energy(atoms)
        result["e_adsorbate"] = e_adsorbate

        calc = FAIRChemCalculator(batch_predict_unit, task_name="oc20")

        # Relax system
        system.calc = calc
        opt_sys = LBFGS(system, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_system = opt_sys.run(fmax=0.05, steps=500)
        steps_system = opt_sys.get_number_of_steps()
        e_sys_relaxed = system.get_potential_energy()

        # Relax slab
        slab.calc = calc
        opt_slab = LBFGS(slab, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_slab = opt_slab.run(fmax=0.05, steps=500)
        steps_slab = opt_slab.get_number_of_steps()
        e_slab_relaxed = slab.get_potential_energy()

        e_ads = e_sys_relaxed - e_slab_relaxed - e_adsorbate

        result.update({
            "e_sys_relaxed":    e_sys_relaxed,
            "e_slab_relaxed":   e_slab_relaxed,
            "converged_system": bool(converged_system),
            "converged_slab":   bool(converged_slab),
            "steps_system":     steps_system,
            "steps_slab":       steps_slab,
        })

        if abs(e_ads) > 10.0:  # energy_cutoff — hardcoded to match CatFlow convention
            result["error"] = f"|ΔE_ads|={abs(e_ads):.3f} eV exceeds cutoff"
            return result

        diff = abs(e_ads - target_E) if target_E is not None else None
        result["computed_binding_E"] = e_ads
        result["binding_E_diff"] = diff
        result.update(_threshold_flags(diff))

        # Write relaxed structure
        subdir = _threshold_subdir(diff)
        out_path = pathlib.Path(output_dir) / subdir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        system.calc = None
        system.info["computed_binding_E"] = e_ads
        if diff is not None:
            system.info["binding_E_diff"] = diff
        system.write(str(out_path))

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# GPU worker process
# ---------------------------------------------------------------------------

def run_gpu_process(gpu_id: int, tasks: list, num_workers: int) -> list:
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate import InferenceBatcher

    try:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = "cuda"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = "cpu"

        predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
        batcher = InferenceBatcher(
            predict_unit,
            concurrency_backend_options={"max_workers": num_workers},
        )
        process_fn = functools.partial(
            process_single_traj,
            batch_predict_unit=batcher.batch_predict_unit,
        )
        return list(batcher.executor.map(process_fn, tasks))

    except Exception:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    diffs = [float(r["binding_E_diff"]) for r in rows if r.get("binding_E_diff") is not None]
    if not diffs:
        return {}
    counts = {}
    for m in _THRESHOLD_MULTIPLIERS:
        key = f"within_{m}T"
        flagged = [r[key] for r in rows if r.get(key) is not None]
        counts[key] = int(sum(flagged))
    return {
        "n_relaxed": len(diffs),
        "n_total":   len(rows),
        "mean":      statistics.mean(diffs),
        "median":    statistics.median(diffs),
        "std":       statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        "min":       min(diffs),
        "max":       max(diffs),
        **counts,
    }


def format_report(directory: str, stats: dict, n_strucs: int | None, seed: int) -> str:
    from datetime import datetime
    lines = [
        "=" * 52,
        "GACE Traj Relaxation Report  (UMA)",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Directory : {directory}",
        "=" * 52,
    ]
    if not stats:
        lines.append("\nNo relaxed structures with valid binding energies.")
    else:
        lines += [
            "",
            f"Binding Energy  ({stats['n_relaxed']} / {stats['n_total']} processed)",
            f"  Mean |ΔE|   : {stats['mean']:.4f} eV",
            f"  Median |ΔE| : {stats['median']:.4f} eV",
            f"  Std |ΔE|    : {stats['std']:.4f} eV",
            f"  Min |ΔE|    : {stats['min']:.4f} eV",
            f"  Max |ΔE|    : {stats['max']:.4f} eV",
            "",
            "  Within threshold (T = 0.10 eV)",
            f"  ≤  1×T (0.10 eV) : {stats['within_1T']:>5d}",
            f"  ≤  2×T (0.20 eV) : {stats['within_2T']:>5d}",
            f"  ≤  3×T (0.30 eV) : {stats['within_3T']:>5d}",
            f"  ≤  5×T (0.50 eV) : {stats['within_5T']:>5d}",
            f"  ≤ 10×T (1.00 eV) : {stats['within_10T']:>5d}",
        ]
    if n_strucs is not None:
        lines.append(f"\n  (sampled {n_strucs} structures, seed={seed})")
    lines += ["", "=" * 52]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

class CsvWriter:
    def __init__(self, path: str, fieldnames: list[str]):
        self._path = path
        self._fieldnames = fieldnames
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            self._writer.writeheader()

    def write(self, row: dict):
        self._writer.writerow({k: _csv_fmt(row.get(k)) for k in self._fieldnames})
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Relax pre-sanitized .traj structures with UMA and compute adsorption energies.")
    p.add_argument("--strucs_dir",     required=True,  help="Root directory to search for .traj files")
    p.add_argument("--relaxed_strucs", default="./relaxed_strucs")
    p.add_argument("--uma_model",      default="uma-s-1p1")
    p.add_argument("--fmax",           type=float, default=0.05)
    p.add_argument("--steps",          type=int,   default=500)
    p.add_argument("--energy_cutoff",  type=float, default=10.0)
    p.add_argument("--n_strucs",       type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--num_workers",    type=int,   default=16, help="Concurrent relaxations per GPU process")
    p.add_argument("--num_gpus",       type=int,   default=None, help="GPUs to use (default: all available)")
    p.add_argument("--cpu",            action="store_true")
    p.add_argument("--output",         default="./uma_report.txt")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # Collect files
    files = sorted(str(p) for p in pathlib.Path(args.strucs_dir).rglob("*.traj"))
    log.info(f"Found {len(files)} .traj files under {args.strucs_dir}")
    if not files:
        raise FileNotFoundError(f"No .traj files found under {args.strucs_dir}")

    if args.n_strucs is not None and args.n_strucs < len(files):
        files = sorted(random.Random(args.seed).sample(files, args.n_strucs))
        log.info(f"Sampled {len(files)} structures (seed={args.seed})")

    # Read target energies from each file (fast, no calculator needed)
    log.info("Reading target binding energies from atoms.info...")
    target_energies: dict[str, float | None] = {}
    for f in files:
        try:
            atoms = read(f)
            target_energies[f] = atoms.info.get("target_binding_E")
        except Exception as exc:
            log.warning(f"Could not read {f}: {exc}")
            target_energies[f] = None

    relaxed_dir = pathlib.Path(args.relaxed_strucs)
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = [(f, target_energies[f], str(relaxed_dir)) for f in files]

    # GPU / CPU dispatch
    num_gpus = 0
    if not args.cpu and torch.cuda.is_available():
        num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()

    all_results: list[dict] = []

    if num_gpus > 0:
        log.info(f"Using {num_gpus} GPU(s), {args.num_workers} concurrent relaxations each")
        tasks_per_gpu = [[] for _ in range(num_gpus)]
        for i, task in enumerate(tasks):
            tasks_per_gpu[i % num_gpus].append(task)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_gpus) as pool:
            async_results = [
                pool.apply_async(run_gpu_process, (gpu_id, tasks_per_gpu[gpu_id], args.num_workers))
                for gpu_id in range(num_gpus)
                if tasks_per_gpu[gpu_id]
            ]
            for res in tqdm(async_results, desc="GPU batches"):
                all_results.extend(res.get())
    else:
        log.info(f"Using CPU, {args.num_workers} concurrent relaxations")
        all_results = run_gpu_process(-1, tasks, args.num_workers)

    # Write CSV
    csv_path = str(relaxed_dir / "relaxed_strucs_data.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        writer = CsvWriter(csv_path, fieldnames)
        for row in all_results:
            writer.write(row)
        writer.close()
        log.info(f"CSV written to {csv_path}")

    # Convergence summary
    n_total = len(all_results)
    n_conv_sys  = sum(1 for r in all_results if r.get("converged_system"))
    n_conv_slab = sum(1 for r in all_results if r.get("converged_slab"))
    n_errors    = sum(1 for r in all_results if r.get("error"))
    log.info(f"System converged: {n_conv_sys}/{n_total}  Slab converged: {n_conv_slab}/{n_total}  Errors: {n_errors}")

    stats = summarize(all_results)
    report = format_report(args.strucs_dir, stats, args.n_strucs, args.seed)
    print(report)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    log.info(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
