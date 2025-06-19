#!/bin/bash
#SBATCH -A chm250007p #project number
#SBATCH -N 1
#SBATCH -p RM
#SBATCH --ntasks-per-node=10
#SBATCH -t 00:30:00
#SBATCH -J vac_spc_ #job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmsween@umich.edu

source ~/.bashrc 

conda activate /ocean/projects/chm250007p/dsweeney/.envs/cfm
pip install --user tview
python3 -X importtime -c "import torch; import torch_scatter; import your_package" 2> import.log
python3 -m tview import.log
