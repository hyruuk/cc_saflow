#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=24:00:00
#SBATCH --job-name=safFeatures
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

$HOME/python_envs/saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/features/compute_$1.py -nt 8 -s $2 -r $3 -t alltrials
