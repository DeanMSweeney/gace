"""Evaluate forge-generated structures for uniqueness, validity, and binding energy.

All logic lives in gace.utils.evaluation. This script is the CLI entry point.

Arguments
---------
--gen_strucs       Directory containing generated structure files (.vasp preferred, .traj fallback).
                   Default: ./gen_strucs

--sanitize         Flag. If set, runs CatalystForge validity checks on structures and reports
                   per-criterion pass rates (vol_ok, cat_dist_ok, etc.). Disabled by default.

--relax            Flag. If set, runs MLFF relaxation on structures (implies --sanitize) and
                   reports binding energy statistics. Disabled by default.

--rlxd_strucs      Output directory for relaxed .vasp files and evaluation CSVs.
                   Default: ./rlxd_strucs

--n_strucs         Number of structures to randomly sample for sanitisation/relaxation.
                   Default: all structures in gen_strucs.

--mlff_checkpoint  Name of the fairchem MLFF model to use for relaxation.
                   Default: EquiformerV2-31M-S2EF-OC20-All+MD

--energy_cutoff    Maximum allowed computed binding energy (eV). Structures above this
                   threshold are excluded from relaxation results. Default: 10.0

--fmax             Force convergence criterion for the MLFF relaxation (eV/Å). Default: 0.05

--steps            Maximum number of optimizer steps per structure. Default: 200

--cpu              Flag. Forces MLFF to run on CPU instead of GPU.

--seed             Random seed for reproducible structure sampling. Default: 42

--num_workers      Number of parallel workers for structure loading during uniqueness
                   computation. Default: 8

--output           Path to write the evaluation report (.txt).
                   Default: ./eval_report.txt

Examples
--------
    # Uniqueness only (fast):
    python scripts/evaluate.py --gen_strucs ./gen_strucs

    # Uniqueness + validity checks:
    python scripts/evaluate.py --gen_strucs ./gen_strucs --sanitize

    # With MLFF relaxation on a 500-structure sample:
    python scripts/evaluate.py --gen_strucs ./gen_strucs --relax --n_strucs 500

    # Full run, all structures, save report to custom path:
    python scripts/evaluate.py --gen_strucs ./gen_strucs --relax --output ./results/eval.txt
"""

from gace.utils.posthoc_evaluation import main as evaluate_generated_structures

if __name__ == "__main__":
    evaluate_generated_structures()
