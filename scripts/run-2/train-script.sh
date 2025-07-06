#!/bin/bash
#SBATCH -A chm250007p                    # project number
#SBATCH -p GPU-shared
#SBATCH -t 15:00:00
#SBATCH --gpus=v100-32:4
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

PROJECT_DIR=$(dirname "$(dirname "$SLURM_SUBMIT_DIR")")

srun singularity exec --nv --cleanenv \
    --bind "${PROJECT_DIR}":/workspace \
    "/ocean/projects/chm250007p/dsweeney/adsflow.sif" \
    bash -c "\
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
      cd /workspace/scripts && \
      echo \"NOW IN: \$(pwd)\" && \
      echo \"CONTENTS:\" && ls -l . && \
      python train-gpu.py
    "

