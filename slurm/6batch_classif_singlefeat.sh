for RUN in 0 2 3 4 5 6 7; do
  sbatch ./slurm/6_classif_singlefeat.sh VTC 50 50 group 04 RUN
  sbatch ./slurm/6_classif_singlefeat.sh VTC 25 75 group 04 RUN
  sbatch ./slurm/6_classif_singlefeat.sh VTC 10 90 group 04 RUN
  
  sbatch ./slurm/6_classif_singlefeat.sh odd 50 50 group 04 RUN

  for SUBJ in 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 28 29 30 31 32 33 34 35 36 37 38; do
    sbatch ./slurm/6_classif_singlefeat.sh VTC 50 50 subject $SUBJ RUN
    sbatch ./slurm/6_classif_singlefeat.sh VTC 25 75 subject $SUBJ RUN
    sbatch ./slurm/6_classif_singlefeat.sh VTC 10 90 subject $SUBJ RUN

    sbatch ./slurm/6_classif_singlefeat.sh odd 50 50 subject $SUBJ RUN
  done
done
