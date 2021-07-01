##### OPEN PREPROC FILES AND SEGMENT THEM
from src.neuro import split_PSD_data, split_trials
from src.saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH, BIDS_PATH, LOGS_DIR
from src.utils import get_SAflow_bids
from scipy.io import savemat
import pickle

if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for run in BLOCS_LIST:

            INepochs, OUTepochs = split_trials(BIDS_PATH, LOGS_DIR, subj=subj, run=run, stage='PSD', by='VTC')
            INepochs_path, INepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='IN')
            OUTepochs_path, OUTepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='OUT')
            
            with open(INepochs_filename, 'wb') as fp:
                pickle.dump(INepochs, fp)

            with open(OUTepochs_filename, 'wb') as fp:
                pickle.dump(OUTepochs, fp)