#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=safSegmentation
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24


$HOME/python_envs/saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/source_reconstruction/inverse_solution.py -s $1 -r $2
