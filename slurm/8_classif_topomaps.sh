#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=1:00:00
#SBATCH --job-name=safClassifTopomaps
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/visualization/classif_topomaps.py -n $1
