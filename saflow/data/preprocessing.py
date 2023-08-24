import os
import os.path as op
#from saflow.utils import get_SAflow_bids
from saflow.neuro import saflow_preproc
from saflow import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import argparse
from mne_bids import BIDSPath

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
		file_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='raw')[1]
		save_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_raw')[1]
		report_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_report')[1]

		input_path = BIDSPath(subject=subj,
					task="gradCPT",
					run='0'+bloc,
					datatype="meg",
					extension=".fif",
					root=BIDS_PATH)
		output_path = BIDSPath(subject=subj,
			task="gradCPT",
			run='0'+bloc,
			datatype="meg",
			description='cleaned',
			extension=".fif",
			root=BIDS_PATH)
		report_path = BIDSPath(subject=subj,
			task="gradCPT",
			run='0'+bloc,
			datatype="report",
			extension=".html",
			root=BIDS_PATH)
		saflow_preproc(input_path, output_path, report_path, ica=ica)
