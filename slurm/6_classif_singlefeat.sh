#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=00:30:00
#SBATCH --job-name=safSingleFeature
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/models/run_classifs.py -m LDA -mf 0 -c $1 -by VTC -s 50 50 -avg 0 -n 1 -r all
