#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=saflow_generate_bids
#SBATCH --mem=512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24


$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/src/features/split_trials.py -by odd
