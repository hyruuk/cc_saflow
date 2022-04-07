#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=safMultiFeatures
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/models/run_classifs.py -mf 1 -m LR -c $1 -by VTC -s 50 50 -avg 0 
