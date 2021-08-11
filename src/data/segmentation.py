##### OPEN PREPROC FILES AND SEGMENT THEM
from src.utils import get_SAflow_bids
from src.neuro import segment_files
from src.saflow_params import BIDS_PATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST
import pickle
import os

if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            preproc_path, preproc_filename = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='preproc_raw', cond=None)
            epoched_path, epoched_filename = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='-epo', cond=None)
            ARlog_path, ARlog_filename = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='ARlog', cond=None)
            if not os.path.isfile(epoched_filename):
                epochs_clean, AR_log = segment_files(preproc_filename, tmin=0, tmax=0.8)
                epochs_clean.save(epoched_filename, overwrite=True)
                del epochs_clean
                with open(ARlog_filename, 'wb') as fp:
                    pickle.dump(AR_log, fp)
            else:
                print('{} {} File already exists'.format(subj, bloc))
