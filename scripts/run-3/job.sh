#!/bin/bash
#SBATCH -A chm250007p                      # your project/account
#SBATCH -p GPU-shared                      # GPU partition
#SBATCH --nodes=1                          # 1 node
#SBATCH --ntasks-per-node=1                # one process per node
#SBATCH --cpus-per-task=20                  # CPUs per task
#SBATCH --gres=gpu:v100-32:4               # 1 × V100-32 GB GPU
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
PROJECT_DIR=$(dirname "$(dirname "$SLURM_SUBMIT_DIR")")

# Run command
srun singularity exec --nv --cleanenv \
    --bind "${PROJECT_DIR}":/workspace \
    "$SCRATCH/.imgs/adsflow.sif" \
    bash -c "\
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      python train.py \
        devices=4 \
        num_nodes=1 \
	num_workers=4 \
        accelerator='gpu' \
	strategy='ddp' \
    "
