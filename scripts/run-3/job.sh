#!/bin/bash
#SBATCH -A chm250007p                    # project number
#SBATCH -p GPU-shared
#SBATCH -t 15:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -J adsflow_train-0               # job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


module purge
module load cuda/12.4

# point to the project root (for your singularity bind)
PROJECT_DIR=$(dirname "$(dirname "$SLURM_SUBMIT_DIR")")

srun singularity exec --nv --cleanenv \
    --bind "${PROJECT_DIR}":/workspace \
    "$SCRATCH/.imgs/adsflow.sif" \
    bash -c "\
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      python train.py
    "
