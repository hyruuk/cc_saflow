for SUB in {0..270}; do
  sbatch ./slurm/3bis_computePSD.sh $SUB
done
