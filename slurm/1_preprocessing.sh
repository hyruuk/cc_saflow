#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=safPreproc
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12



$HOME/python_envs/saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/data/preprocessing.py -s $1
