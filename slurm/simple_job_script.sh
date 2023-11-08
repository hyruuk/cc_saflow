#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=12:00:00
#SBATCH --job-name=saf_simple
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

$HOME/python_envs/saflow/bin/python $1
