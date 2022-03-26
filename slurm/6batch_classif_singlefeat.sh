for BAND in delta theta alpha lobeta hibeta gamma1 gamma2 gamma3; do
for CHAN in {0..270}; do
  sbatch ./slurm/6_classif_singlefeat.sh $CHAN $BAND
done
done
