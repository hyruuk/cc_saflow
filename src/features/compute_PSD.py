import mne
import numpy as np
from src.utils import get_SAflow_bids
from src.neuro import compute_PSD
from src.saflow_params import FOLDERPATH, IMG_DIR, FREQS, SUBJ_LIST, BLOCS_LIST
from scipy.io import savemat

### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='epo', cond=None)
            data = mne.read_epochs(SAflow_bidspath)
            psds = compute_PSD(data, data.info['sfreq'], epochs_length = 0.8, f=FREQS)
            PSD_bidsname, PSD_bidspath = get_SAflow_bids(FOLDERPATH, subj, bloc, stage='PSD', cond=None)
            savemat(PSD_bidspath, {'PSD': psds})
