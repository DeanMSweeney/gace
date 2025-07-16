#!/bin/bash
#SBATCH -A chm250007p                      # your project/account
#SBATCH -p RM-shared                       # GPU partition
#SBATCH --nodes=1                          # 1 node
#SBATCH --ntasks-per-node=8                # one process per node
#SBATCH --cpus-per-task=5                  # 5 cpus per process
#SBATCH -t 22:00:00                        # walltime
#SBATCH -J adsflow_train-0                 # job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Clear environment and load CUDA
module purge
module load cuda/12.4

# Memory management (good to keep)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Point to project root (for singularity bind)
PROJECT_DIR=$(dirname "$SLURM_SUBMIT_DIR")

# Run command
srun singularity exec --cleanenv \
    --bind "${PROJECT_DIR}":/workspace \
    "$SCRATCH/.imgs/adsflow.sif" \
    bash -c "\
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      python debug.py \
    "
