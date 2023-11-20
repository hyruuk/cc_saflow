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
import os.path as op
#import antropy


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='all',
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-r",
    "--run",
    default='all',
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
parser.add_argument(
    "-l",
    "--level",
    default="sensor",
    type=str,
    help="Level of processing (sensor or source)",
)

def compute_lzc_for_epoch(epoch, idx, events_dict, filepaths):
    fname = str(filepaths['lzc'].fpath).replace('idx', str(idx)) + '.pkl'
    if not os.path.exists(fname):
        print(f'Epoch {idx}')
        n_jobs = -1  # Uses all processors. Adjust if needed.
        lzc_array = Parallel(n_jobs=n_jobs)(delayed(compute_lzc_for_chan)(channel, chan_idx) 
                                            for chan_idx, channel in enumerate(epoch))
        epoch_array = np.array(lzc_array).T
        # save
        with open(fname, 'wb') as f:
            pickle.dump({'data':epoch_array,
                         'info':events_dict}, f)
        print(f'Saved {fname}')
    # if exists, just load
    else:
        with open(fname, 'rb') as f:
            file = pickle.load(f)
            epoch_array = file['data']

    return epoch_array

def compute_lzc_for_chan(channel, chan_idx):
    # Compute LZC and permuted LZC
    plzc = complexity_lempelziv(channel, permutation=True, dimension=7, delay=2)[0]
    nk_lzc = complexity_lempelziv(channel, symbolize='median', permutation=False)[0]
    #channel_binarized = np.array([0 if x < np.median(channel) else 1 for x in channel])
    #ant_lzc = antropy.lziv_complexity(channel_binarized, normalize=True)
    print(f'Chan : {chan_idx}, LZC = {ant_lzc}')
    return [nk_lzc, plzc]


def compute_LZC_on_sources(data, sfreq, filepaths, n_trials=8):
    # Segment array
    segmented_array, events_idx, events_dicts = segment_sourcelevel(data, filepaths, sfreq=sfreq, n_events_window=n_trials)
    
    for epo_idx, epoch in enumerate(segmented_array):
        lzc_array = compute_lzc_for_epoch(epoch, events_idx[epo_idx], events_dicts[epo_idx], filepaths)
    
    lzc_array = np.array(lzc_array)
    return lzc_array, events_idx, events_dicts


if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    run = args.run
    n_trials = args.n_trials
    level = args.level
    freq_bands = saflow.FREQS
    freq_names = saflow.FREQS_NAMES

    if type(run) == str:
        if run == 'all':
            runs = ['02', '03', '04', '05', '06', '07']
        else:
            runs = [run]
    elif type(run) == list:
        runs = run

    if type(subject) == str:
        if subject == 'all':
            subjects = saflow.SUBJ_LIST
        else:
            subjects = [subject]

    for subject in subjects:
        for run in runs:
            print(f'Processing subject {subject}, run {run}')
            filepaths = create_fnames(subject, run)
            filepaths['lzc'].update(root=op.join('/'.join(str(filepaths['lzc'].root).split('/')[:-1]), str(filepaths['lzc'].root).split('/')[-1] + f'_{level}_{n_trials}trials_noplzc_ant'))
            filepaths['lzc'].mkdir(exist_ok=True)
            if level == 'sensor':
                raw = mne_bids.read_raw_bids(filepaths['preproc'])
                picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)
                data = raw.get_data(picks=picks)
                sfreq = raw.info['sfreq']
            elif level == 'source':
                stc = mne.read_source_estimate(filepaths['morph'])
                data = np.float64(stc.data)
                sfreq = stc.sfreq

            lzc_array, events_idx, events_dicts = compute_LZC_on_sources(data, sfreq, filepaths, n_trials=n_trials)