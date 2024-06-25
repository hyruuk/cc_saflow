#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --time=00:30:00
#SBATCH --job-name=safAtlas
#SBATCH --mem=64G
#SBATCH --ntasks=1

$HOME/python_envs/saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/source_reconstruction/apply_atlas.py -s $1 -r $2
