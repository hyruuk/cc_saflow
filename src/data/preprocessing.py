import os
import os.path as op
from src.utils import get_SAflow_bids
from src.neuro import saflow_preproc, find_rawfile
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, REPORTS_PATH


if __name__ == "__main__":
	# create report path
	try:
		os.mkdir(REPORTS_PATH)
	except:
		print('Report path already exists.')
	for subj in SUBJ_LIST:
		for bloc in BLOCS_LIST:
			filepath, filename = find_rawfile(subj, bloc, FOLDERPATH)
			save_pattern = get_SAflow_bids(FOLDERPATH, subj=subj, run=bloc, stage='preproc_raw')[1]
			#save_pattern =  op.join(FOLDERPATH + filepath, filename[:-3] + '_preproc_raw.fif.gz')
			report_pattern = op.join(REPORTS_PATH, filename[:-3] + '_preproc_report.html')
			full_filepath = FOLDERPATH + filepath + filename
			saflow_preproc(full_filepath, save_pattern, report_pattern)
