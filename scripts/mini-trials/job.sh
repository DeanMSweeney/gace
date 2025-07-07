#!/bin/bash
#SBATCH -A chm250007p                    # project number
#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH --ntasks=1                       # p processes
#SBATCH --cpus-per-task=16               # n cores
#SBATCH -J adsflow_train-0               # job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge

# point to the project root (for your singularity bind)
PROJECT_DIR=$(dirname "$(dirname "$SLURM_SUBMIT_DIR")")

srun singularity exec --cleanenv \
    --bind "${PROJECT_DIR}":/workspace \
    "$SCRATCH/.imgs/adsflow.sif" \
    bash -c "\
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      python train-cpu.py
    "

