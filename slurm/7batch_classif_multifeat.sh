for MODEL in SVM LR RF; do
for FREQ in delta theta alpha lobeta hibeta gamma1 gamma2 gamma3; do
  sbatch ./slurm/7_classif_multifeat.sh $FREQ $MODEL
done
done
