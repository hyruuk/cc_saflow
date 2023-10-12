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
#from neurokit2.complexity import complexity_lempelziv, complexity_dimension, complexity_delay
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


def compute_hilbert_env(stc, l_freq, h_freq):
    """
    Compute the Hilbert envelope of a source estimate object, after band-pass filtering it between l_freq and h_freq.

    Parameters
    ----------
    stc : mne.SourceEstimate
        The source estimate object to compute the Hilbert envelope of.
    l_freq : float
        The lower frequency of the band-pass filter.
    h_freq : float
        The higher frequency of the band-pass filter.

    Returns
    -------
    stc_hilb : np.ndarray
        The Hilbert envelope of the source estimate object, with shape (n_vertices, n_samples).
    """
    stc_filtered = mne.filter.filter_data(np.float64(stc.data), sfreq=stc.sfreq, l_freq=l_freq, h_freq=h_freq)
    stc_hilb = np.abs(hilbert(stc_filtered))**2
    return stc_hilb



def time_average(segmented_array):
    """
    Average the time dimension of a segmented data array.

    Parameters
    ----------
    segmented_array : np.ndarray
        The segmented data array, with shape (n_events, n_channels, n_samples).

    Returns
    -------
    time_avg_array : np.ndarray
        The time-averaged data array, with shape (n_events, n_channels).
    """
    time_avg_array = []
    for array in segmented_array:
        time_avg_array.append(np.mean(array, axis=1))
    time_avg_array = np.array(time_avg_array)
    return time_avg_array

def compute_PSD(stc, filepaths):
    """
    Compute the power spectral density (PSD) of a source estimate object across different frequency bands.

    Parameters
    ----------
    stc : mne.SourceEstimate
        The source estimate object to compute the PSD of.

    Returns
    -------
    psd_array : np.ndarray
        The PSD of the source estimate object, with shape (n_freq_bands, n_events, n_channels).
    """
    psd_array = []
    for idx, freq in enumerate(saflow.FREQS_NAMES):
        stc_env = compute_hilbert_env(stc, saflow.FREQS[idx][0], saflow.FREQS[idx][1])
        segmented_array, events_idx, events_dicts = segment_sourcelevel(stc_env, filepaths, sfreq=stc.sfreq)
        time_avg_array = time_average(segmented_array)
        psd_array.append(time_avg_array)
    psd_array = np.array(psd_array).transpose(1,0,2)

    return psd_array, events_idx, events_dicts


if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    run = args.run
    freq_bands = saflow.FREQS
    freq_names = saflow.FREQS_NAMES

    filepaths = create_fnames(subject, run)

    stc = mne.read_source_estimate(filepaths['morph'])

    psd_array, events_idx, events_dicts = compute_PSD(stc, filepaths)

    for idx, array in enumerate(psd_array):
        fname = str(filepaths['psd'].fpath).replace('idx', str(events_idx[idx])) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(array, f)