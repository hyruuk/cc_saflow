for RUN in 0 2 3 4 5 6 7; do
for CHAN in {0..270}; do
  sbatch ./slurm/6_classif_singlefeat.sh $CHAN $RUN
done
done
