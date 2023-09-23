#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=2:00:00
#SBATCH --job-name=safInverse
#SBATCH --mem=256G
#SBATCH --ntasks=1

$HOME/python_envs/saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/source_reconstruction/inverse_solution.py -s $1 -r $2
