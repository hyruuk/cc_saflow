  #!/bin/bash
  #SBATCH --account=def-kjerbi
  #SBATCH --time=12:00:00
  #SBATCH --job-name=saflow_generate_bids
  #SBATCH --mem=31G
  #SBATCH --nodes=1
  #SBATCH --ntasks-per-node=12

for SUB in 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25; do
  module load python/3.7.0

  $HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/src/data/preprocessing.py -s $SUB
done
