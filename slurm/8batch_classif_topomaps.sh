for RUN in 2 3 4 5 6 7 allruns; do
  sbatch slurm/8_classif_topomaps.sh VTC_PSD_LDA_group-level_singlefeat_single-trial_normalized_1000perm_5050-split_run-$RUN
done
