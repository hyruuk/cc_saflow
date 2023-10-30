import argparse
import saflow
import mne
from mne_bids import BIDSPath, read_raw_bids
import mne
import saflow
import os
import numpy as np
import pandas as pd
import mne_bids
from scipy.signal import hilbert
from mne_bids import BIDSPath, read_raw_bids
from neurokit2.complexity import complexity_lempelziv, complexity_dimension, complexity_delay
import pickle
from joblib import Parallel, delayed
import numpy as np
import pickle
from saflow.features.utils import create_fnames, segment_sourcelevel


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='12',
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-r",
    "--run",
    default='02',
    type=str,
    help="Run to process",
)
parser.add_argument(
    "-nt",
    "--n_trials",
    default=8,
    type=int,
    help="Number of trials to consider per epoch",
)

def compute_lzc_for_epoch(epoch, idx, filepaths):
    fname = str(filepaths['lzc'].fpath).replace('idx', str(idx)) + '.pkl'
    if not os.path.exists(fname):
        print(f'Epoch {idx}')
        n_jobs = -1  # Uses all processors. Adjust if needed.
        lzc_array = Parallel(n_jobs=n_jobs)(delayed(compute_lzc_for_chan)(channel, chan_idx) 
                                            for chan_idx, channel in enumerate(epoch))
        epoch_array = np.array(lzc_array)
        # save
        with open(fname, 'wb') as f:
            pickle.dump({'data':epoch_array.T,
                         'info':events_dicts[idx]}, f)
    # if exists, just load
    else:
        with open(fname, 'rb') as f:
            file = pickle.load(f)
            epoch_array = file['data']
    return epoch_array

def compute_lzc_for_chan(channel, chan_idx):
    # Compute LZC and permuted LZC
    print(f'Channel {chan_idx}')
    plzc = complexity_lempelziv(channel, permutation=True, dimension=7, delay=2)[0]
    lzc = complexity_lempelziv(channel, permutation=False)[0]
    return [lzc, plzc]


def compute_LZC_on_sources(stc, filepaths, n_trials=8):
    # Segment array
    segmented_array, events_idx, events_dicts = segment_sourcelevel(stc.data, filepaths, sfreq=stc.sfreq, n_events_window=n_trials)
    
    for epo_idx, epoch in enumerate(segmented_array):
        lzc_array = compute_lzc_for_epoch(epoch, epo_idx, filepaths)
    
    lzc_array = np.array(lzc_array)
    return lzc_array, events_idx, events_dicts


if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    run = args.run
    n_trials = args.n_trials
    freq_bands = saflow.FREQS
    freq_names = saflow.FREQS_NAMES

    filepaths = create_fnames(subject, run)

    stc = mne.read_source_estimate(filepaths['morph'])

    lzc_array, events_idx, events_dicts = compute_LZC_on_sources(stc, filepaths, n_trials=n_trials)