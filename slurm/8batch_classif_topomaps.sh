for RUN in 2 3 4 5 6 7 allruns; do
  $HOME/python_envs/cc_saflow/bin/python $HOME/projects/def-kjerbi/hyruuk/cc_saflow/saflow/visualization/classif_topomaps.py -n VTC_PSD_LDA_group-level_singlefeat_single-trial_normalized_1000perm_5050-split_run-$RUN
done
