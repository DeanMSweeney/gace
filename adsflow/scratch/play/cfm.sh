#!/bin/bash
#SBATCH -A chm250007p             # project number
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=64      # 10 tasks (or threads) if you really need them
#SBATCH -p RM-shared              # partition
#SBATCH -t 01:00:00               # walltime
#SBATCH -J vac_spc_               # job name
#SBATCH --mail-type=ALL           # email notifications
#SBATCH --mail-user=dmsween@umich.edu

# make sure 'conda activate' works in a non-interactive shell
source ~/.bashrc

# activate your environment
mamba activate cfm-sonic
mamba info

python -u scratch.py 
