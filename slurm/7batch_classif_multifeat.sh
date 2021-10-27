for SUB in {0..270}; do
  sbatch ./slurm/7_classif_multifeat.sh $SUB
done
