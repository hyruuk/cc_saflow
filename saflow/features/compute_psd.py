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
import os.path as op

parser = argparse.ArgumentParser()

parser.add_argument(
    "-nt",
    "--n_trials",
    default=1,
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
parser.add_argument(
    "-m",
    "--method",
    default="hilbert",
    type=str,
    help="Level of processing (sensor or source)",
)
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


def compute_hilbert_env(data, sfreq, l_freq, h_freq):
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
    stc_filtered = mne.filter.filter_data(data, 
                                          sfreq=sfreq, 
                                          l_freq=l_freq, 
                                          h_freq=h_freq, 
                                          n_jobs=-1)
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

def compute_PSD(data, filepaths, sfreq, n_trials=1, method='welch'):
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
    if method == 'hilbert':
        psd_array = []
        for idx, freq in enumerate(saflow.FREQS_NAMES):
            stc_env = compute_hilbert_env(data, sfreq, saflow.FREQS[idx][0], saflow.FREQS[idx][1])
            segmented_array, events_idx, events_dicts = segment_sourcelevel(stc_env, filepaths, sfreq=sfreq, n_events_window=n_trials)
            time_avg_array = time_average(segmented_array)
            psd_array.append(time_avg_array)
        psd_array = np.array(psd_array).transpose(1,0,2)
    elif method == 'welch':
        psd_array = []
        segmented_array, events_idx, events_dicts = segment_sourcelevel(data, filepaths, sfreq=sfreq, n_events_window=n_trials)
        welch_array = mne.time_frequency.psd_array_welch(segmented_array, sfreq=sfreq, n_jobs=-1, n_fft=1022)#fmin=saflow.FREQS[idx][0], fmax=saflow.FREQS[idx][1])
        for idx, freq in enumerate(saflow.FREQS_NAMES):

            lowfreq = saflow.FREQS[idx][0]
            highfreq = saflow.FREQS[idx][1]
            freq_mask = np.where((welch_array[1] >= lowfreq) & (welch_array[1] <= highfreq))[0]

            if len(freq_mask) <= 1:
                welch_array_masked = welch_array[0][:,:,freq_mask]
            else:
                welch_array_masked = welch_array[0][:,:,freq_mask[0]:freq_mask[-1]+1]
            psd_array.append(np.mean(welch_array_masked, axis=-1))
        psd_array = np.array(psd_array).transpose(1,0,2)
        
    return psd_array, events_idx, events_dicts


if __name__ == "__main__":
    args = parser.parse_args()
    n_trials = args.n_trials
    subject = args.subject
    run = args.run
    level = args.level
    method = args.method
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
            filepaths = create_fnames(subject, run)
            filepaths['psd'].update(root=op.join('/'.join(str(filepaths['psd'].root).split('/')[:-1]), str(filepaths['psd'].root).split('/')[-1] + f'_{method}_{level}_{n_trials}'))

            filepaths['psd'].mkdir(exist_ok=True)

            if level == 'source':
                stc = mne.read_source_estimate(filepaths['morph'])
                data = np.float64(stc.data)
                sfreq = stc.sfreq
            elif level == 'sensor':
                raw = mne_bids.read_raw_bids(filepaths['preproc'])
                data = raw.get_data()
                sfreq = raw.info['sfreq']

            psd_array, events_idx, events_dicts = compute_PSD(data, filepaths, sfreq, n_trials=n_trials, method=method)

            for idx, array in enumerate(psd_array):
                fname = str(filepaths['psd'].fpath).replace('idx', str(events_idx[idx])) + '.pkl'
                with open(fname, 'wb') as f:
                    pickle.dump({'data':array,
                                'info':events_dicts[idx]}, 
                                f)