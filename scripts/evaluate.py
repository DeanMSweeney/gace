"""Evaluate forge-generated structures for novelty, validity, and binding energy.

All logic lives in gace.utils.posthoc_evaluation. This script is the CLI entry point.

Arguments
---------
--strucs_for_eval  Directory containing generated structure files (.vasp preferred, .traj fallback).

--check_novelty    Flag. If set, runs structural uniqueness check via StructureMatcher and
                   reports total/unique counts and uniqueness score. Default: False.

--sanitize         Flag. If set, runs CatalystForge validity checks on structures and reports
                   per-criterion pass rates (vol_ok, cat_dist_ok, etc.). Default: False.

--relax            Flag. If set, runs MLFF relaxation on structures without requiring sanitization
                   and reports binding energy statistics. Disabled by default.

--relaxed_strucs   Output directory for relaxed .vasp files and evaluation CSVs.
                   Default: ./relaxed_strucs

--sanitized_strucs Output directory for sanitized .vasp files.
                   Default: ./sanitized_strucs

--catalyst_dist_cutoff  Catalyst inter-atom distance cutoff as a fraction of the sum of
                   covalent radii. Default: 0.75

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

--num_workers      Number of parallel workers for structure loading. Default: 8

--output           Path to write the evaluation report (.txt).
                   Default: ./eval_report.txt

Examples
--------
    # Novelty check only (fast):
    python scripts/evaluate.py --strucs_for_eval ./gen_strucs --check_novelty

    # Validity checks only:
    python scripts/evaluate.py --strucs_for_eval ./gen_strucs --sanitize

    # Relax on the fly without sanitization, on a 500-structure sample:
    python scripts/evaluate.py --strucs_for_eval ./gen_strucs --relax --n_strucs 500

    # Full run — novelty + sanitize + relax, save report to custom path:
    python scripts/evaluate.py --strucs_for_eval ./gen_strucs --check_novelty --sanitize --relax --output ./results/eval.txt
"""

from gace.utils.posthoc_evaluation import main as evaluate_generated_structures

if __name__ == "__main__":
    evaluate_generated_structures()
