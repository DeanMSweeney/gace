#!/bin/bash
#SBATCH -A chm250007p
#SBATCH -p RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 10:00:00
#SBATCH -J adsflow_train-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Clear environment and load CUDA
module purge

# Point to project root (for singularity bind)
PROJECT_DIR=$(dirname "$SLURM_SUBMIT_DIR")

# Run command
srun singularity exec \
    --bind "${PROJECT_DIR}":/workspace \
    "$SCRATCH/.imgs/adsflow.sif" \
    bash -c "\
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      export PYTHONUNBUFFERED=1
      python -u gen_vel.py \
    "

