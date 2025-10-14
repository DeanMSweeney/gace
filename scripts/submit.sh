#!/bin/bash
#SBATCH -A chm250015p                      # your project/account
#SBATCH -p GPU                             # GPU partition
#SBATCH --nodes=1                          # 1 node
#SBATCH --ntasks-per-node=8                # one process per node
#SBATCH --gres=gpu:h100-80:8               # 1 × V100-32 GB GPU
#SBATCH --cpus-per-task=12                 # cpus per process 
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

# pick a directory on your workspace or scratch that has enough space
export MY_WANDB_ROOT=/workspace/wandb

# where run metadata & logs go
export WANDB_DIR=$MY_WANDB_ROOT/runs

# where artifacts are staged for upload (the “staging” area)
export WANDB_DATA_DIR=$MY_WANDB_ROOT/staging

# where artifacts you download from the cloud get written
export WANDB_ARTIFACT_DIR=$MY_WANDB_ROOT/artifacts

# (optional) where wandb caches things like artifact definitions
export WANDB_CACHE_DIR=$MY_WANDB_ROOT/cache

# disable active logging 
export WANDB_MODE=offline

# Run command
srun singularity exec --nv \
    --bind "${PROJECT_DIR}":/workspace \
    "$SCRATCH/.imgs/adsflow.sif" \
    bash -c "\
      mkdir -p \
  	$WANDB_DIR \
  	$WANDB_DATA_DIR \
  	$WANDB_ARTIFACT_DIR \
  	$WANDB_CACHE_DIR
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      python train.py \
      --config-path conf \
      --config-name equ_config \
      num_nodes=1 \
      devices=8 \
      logger=True \
    "
