from saflow import SUBJ_LIST, BLOCS_LIST
import os

for subj in SUBJ_LIST:
    for bloc in BLOCS_LIST:
        run = '0' + str(bloc)
        os.system(f'sbatch ./slurm/2_source_reconstruction.sh {subj} {run}')