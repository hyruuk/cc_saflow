#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=saflow_generate_bids
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12


$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/src/features/compute_PSD.py -s $1
