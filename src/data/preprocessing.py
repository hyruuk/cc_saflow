import os
import os.path as op
from src.utils import get_SAflow_bids
from src.neuro import saflow_preproc, find_rawfile
from src.saflow_params import BIDS_PATH, SUBJ_LIST, BLOCS_LIST

if __name__ == "__main__":
	# create report path
	try:
		os.mkdir(REPORTS_PATH)
	except:
		print('Report path already exists.')
	for subj in SUBJ_LIST:
		for bloc in BLOCS_LIST:
			filepath, filename = find_rawfile(subj, bloc, BIDS_PATH)
			save_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_raw')[1]
			report_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_report')[1]
			full_filepath = BIDS_PATH + filepath + filename
			saflow_preproc(full_filepath, save_path, report_path)
