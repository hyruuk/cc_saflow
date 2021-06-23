import mne
import numpy as np
from src.utils import get_SAflow_bids
from src.neuro import compute_PSD
from src.saflow_params import BIDS_PATH, IMG_DIR, FREQS, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST
from scipy.io import savemat

### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='-epo', cond=None)
            data = mne.read_epochs(SAflow_bidspath)
            psds = compute_PSD(data, f=FREQS)
            for psd, freqname in zip(psds, FREQS_NAMES):
                PSD_bidsname, PSD_bidspath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='PSD{}'.format(freqname), cond=None)
                print(PSD_bidspath)
                psd.save(PSD_bidspath, overwrite=True)
            del psds
            del data
