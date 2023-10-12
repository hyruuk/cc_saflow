from saflow import SUBJ_LIST, BLOCS_LIST
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--feature",
    default='psd',
    type=str,
    help="Feature to compute. Can be 'psd', 'lzc', 'slope' or 'conn'.",
)
args = parser.parse_args()
feature = args.feature

for subj in SUBJ_LIST:
    for bloc in BLOCS_LIST:
        run = '0' + str(bloc)
        os.system(f'sbatch ./slurm/3_compute_features.sh {feature} {subj} {run}')