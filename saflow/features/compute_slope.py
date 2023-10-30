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
from mne_bids import BIDSPath, read_raw_bids

import pickle
from fooof import FOOOF
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



def compute_slope(stc, filepaths, fooof_bounds=[5,40], n_trials=8):
    """
    Compute the slope via FOOOF of the source estimate object across different epochs and channels.

    Parameters
    ----------
    stc : mne.SourceEstimate
        The source estimate object to compute the slope of.
    filepaths : dict
        A dictionary containing the filepaths of the preprocessed MEG dataset.

    Returns
    -------
    lzc_array : np.ndarray
        The slope of the source estimate object, with shape (n_epochs, n_channels, 2)
    """
    # Segment array
    segmented_array, events_idx, events_dicts = segment_sourcelevel(stc.data, filepaths, sfreq=stc.sfreq, n_events_window=n_trials)
    slope_array = []
    for idx, epoch in enumerate(segmented_array):
        epoch_array = []
        fname = str(filepaths['slope'].fpath).replace('idx', str(events_idx[idx])) + '.pkl'
        if not os.path.isfile(fname):
            for chan_idx, channel in enumerate(epoch):
                print(f'Epoch {idx} channel {chan_idx}')
                # Prepare FFT and mask
                power_spectrum = abs(np.fft.fft(channel))**2
                freqs = np.fft.fftfreq(channel.size, 1/stc.sfreq)
                fooof_bounds = [5,40]
                mask = np.logical_and(freqs >= fooof_bounds[0], freqs <= fooof_bounds[1])

                fm = FOOOF(max_n_peaks=3)
                fm.fit(freqs[mask], power_spectrum[mask])
                slope = fm.get_params('aperiodic_params')[0]
                epoch_array.append(slope)
            epoch_array = np.array(epoch_array)
            # Save epoch
            with open(fname, 'wb') as f:
                pickle.dump({'data':epoch_array,
                             'info':events_dicts[idx]}, f)
        else:
            with open(fname, 'rb') as f:
                file = pickle.load(f)
                epoch_array = file['data']
        slope_array.append(epoch_array)

    slope_array = np.array(slope_array)
    return slope_array, events_idx

if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    run = args.run
    n_trials = args.n_trials
    freq_bands = saflow.FREQS
    freq_names = saflow.FREQS_NAMES

    filepaths = create_fnames(subject, run)

    stc = mne.read_source_estimate(filepaths['morph'])

    slope_array, events_idx = compute_slope(stc, filepaths, n_trials=n_trials)
