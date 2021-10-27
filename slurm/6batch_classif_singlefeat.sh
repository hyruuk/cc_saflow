for CHAN in {0..270}; do
  sbatch ./slurm/7_classif_multifeat.sh $CHAN
done
