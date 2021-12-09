#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=saflow_SF
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/src/models/classif_singlefeat.py -m LDA -c $1 -f $2 -by resp
