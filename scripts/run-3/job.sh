#!/bin/bash
#SBATCH -A chm250007p                      # your project/account
#SBATCH -p GPU                             # GPU partition
#SBATCH --nodes=2                          # <-- request 2 nodes
#SBATCH --ntasks-per-node=1                # one “task” (process) per node
#SBATCH --cpus-per-task=5                  # CPUs per task
#SBATCH --gpus-per-node=v100-32:8          # 8 × V100-32 GB GPUs on each node
#SBATCH -t 20:00:00                        # walltime
#SBATCH -J adsflow_train-0                 # job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu
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
