for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by VTC -s 50 50 -cor $COR
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by VTC -s 25 75 -cor $COR
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by VTC -s 10 90 -cor $COR
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by odd -s 10 90 -cor $COR
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by VTC -s 50 50 -cor $COR -avg 1
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by VTC -s 25 75 -cor $COR -avg 1
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by VTC -s 10 90 -cor $COR -avg 1
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh -by odd -s 10 90 -cor $COR -avg 1
done
