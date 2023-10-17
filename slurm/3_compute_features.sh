#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=safFeatures
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  

$HOME/python_envs/saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/features/compute_$1.py -nt 8 -s $2 -r $3
