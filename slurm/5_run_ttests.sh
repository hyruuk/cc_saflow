#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=safRunTTests
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24


$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/models/run_ttests.py -by $1 -s $2 $3 -cor $4 -avg $5
