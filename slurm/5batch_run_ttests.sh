for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh VTC 50 50 $COR 0
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh VTC 25 75 $COR 0
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh VTC 10 90 $COR 0
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh odd 10 90 $COR 0
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh VTC 50 50 $COR 1
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh VTC 25 75 $COR 1
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh VTC 10 90 $COR 1
done

for COR in 'maxstat' 'fdr' None; do
sbatch ./slurm/5_run_ttests.sh odd 10 90 $COR 1
done
