#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=saflow_generate_bids
#SBATCH --mem=31G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12

module load python/3.7.0

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/src/data/generate_bids.py
