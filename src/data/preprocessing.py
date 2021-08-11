import os
import os.path as op
from src.utils import get_SAflow_bids
from src.neuro import saflow_preproc, find_rawfile
from src.saflow_params import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='04',
    type=str,
    help="Subject to process",
)
args = parser.parse_args()
if __name__ == "__main__":
	#for subj in SUBJ_LIST:
	subj = args.subject
	for bloc in BLOCS_LIST:
		file_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='raw_ds')[1]
		save_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_raw')[1]
		report_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_report')[1]
		saflow_preproc(file_path, save_path, report_path)
