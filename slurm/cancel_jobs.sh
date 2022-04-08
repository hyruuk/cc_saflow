for JOBID in {$1..$2}; do
scancel JOBID
done
