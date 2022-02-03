import os
import os.path as op
from saflow.utils import get_SAflow_bids
from saflow.neuro import saflow_preproc, find_rawfile
from saflow import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='04',
    type=str,
    help="Subject to process",
)
parser.add_argument(
	"-i",
	"--ica",
	default = True,
	type = bool,
	help="Preprocessing with or without ica"
)
args = parser.parse_args()

if __name__ == "__main__":
	#for subj in SUBJ_LIST:
	subj = args.subject
	ica = args.ica
	for bloc in BLOCS_LIST:
		file_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='raw_ds')[1]
		save_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_raw')[1]
		report_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_report')[1]
		saflow_preproc(file_path, save_path, report_path, ica=ica)
