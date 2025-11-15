#!/bin/bash
#SBATCH --account=bfqy-dtai-gh
#SBATCH --job-name=equiformerv2
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Clear environment and load CUDA
module purge
module load cuda/12.4

# Define work path 
WORK="/projects/bfqy/dsweeney1"

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

# Run command
srun singularity exec --nv \
    --bind "${PROJECT_DIR}":/workspace \
    "$WORK/.containers/adsflow.sif" \
    bash -c "\
      mkdir -p \
  	$WANDB_DIR \
  	$WANDB_DATA_DIR \
  	$WANDB_ARTIFACT_DIR \
  	$WANDB_CACHE_DIR
      cd /workspace${SLURM_SUBMIT_DIR#$PROJECT_DIR} && \
      python train.py \
      --config-path conf \
      --config-name gem_config-32 \
      num_nodes=2 \
      devices=4 \
      logger=True \
    "
