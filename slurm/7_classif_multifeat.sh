#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=safMultiFeatures
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/models/run_classifs.py -mf 1 -m LR -f $1 -by VTC -s 50 50 -avg 0 -n 1
