import mne
import numpy as np
from saflow.utils import get_SAflow_bids
from saflow.neuro import compute_PSD_hilbert, compute_PSD
from saflow.saflow_params import BIDS_PATH, IMG_DIR, FREQS, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST
from scipy.io import savemat
import pickle
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

### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    subj = args.subject
    for bloc in BLOCS_LIST:
        # Generate filenames
        _, epopath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='-epo', cond=None)
        _, rawpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='preproc_raw', cond=None)
        _, ARpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='ARlog', cond=None)
        _, PSDpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='PSD', cond=None)

        # Load files
        epochs = mne.read_epochs(epopath)
        raw = mne.io.read_raw_fif(rawpath, preload=True)
        with open(ARpath, 'rb') as f:
            ARlog = pickle.load(f)

        # Compute envelopes
        envelopes = compute_PSD_hilbert(raw, ARlog, freqlist=FREQS)
        # Save envelopes
        for envelope, freqname in zip(envelopes, FREQS_NAMES):
            _, envpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage='-epoenv_{}'.format(freqname), cond=None)
            envelope.save(envpath, overwrite=True)
        del envelopes

        # Compute PSD
        psds = compute_PSD(epochs, freqlist=FREQS, method='multitaper')
        # Save PSD
        with open(PSDpath, 'wb') as f:
            pickle.dump(psds, f)
        del psds
