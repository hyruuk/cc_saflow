for CHAN in {0..270}; do
  sbatch ./slurm/6_classif_singlefeat.sh $CHAN
done
