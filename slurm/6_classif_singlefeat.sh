#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=48:00:00
#SBATCH --job-name=safSingleFeature
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

$HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/models/run_classifs.py -mf 0 -stage PSD4001200 -by $1 -s $2 $3 -avg 0 -n 1 -l $4 -subj $5 -r $6 -m $7
