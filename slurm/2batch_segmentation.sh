for SUB in 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25; do
  sbatch ./slurm/2_segmentation.sh $SUB
done
