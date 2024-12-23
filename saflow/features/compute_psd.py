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
from fooof import FOOOF, Bands, FOOOFGroup

parser = argparse.ArgumentParser()

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
parser.add_argument(
    "-m",
    "--method",
    default="fooof",
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
parser.add_argument(
    "-j",
    "--n_jobs",
    default=-1,
    type=int,
    help="Number of jobs to use",
)


def compute_hilbert_env(data, sfreq, l_freq, h_freq, n_jobs=-1):
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
                                          n_jobs=n_jobs)
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
        time_avg_array.append(np.nanmean(array, axis=1))
    time_avg_array = np.array(time_avg_array)
    return time_avg_array

def compute_PSD(data, filepaths, sfreq, n_trials=1, method='welch', n_jobs=-1):
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
    if method == 'hilbert':
        for idx, freq in enumerate(saflow.FREQS_NAMES):
            stc_env = compute_hilbert_env(data, sfreq, saflow.FREQS[idx][0], saflow.FREQS[idx][1])
            segmented_array, events_idx, events_dicts = segment_sourcelevel(stc_env, filepaths, sfreq=sfreq, n_events_window=n_trials)

            time_avg_array = time_average(segmented_array)
            psd_array.append(time_avg_array)
        psd_array = np.array(psd_array)
        return psd_array, events_idx, events_dicts

    elif method == 'welch':
        segmented_array, events_idx, events_dicts = segment_sourcelevel(data, filepaths, sfreq=sfreq, n_events_window=n_trials)

        welch_array = mne.time_frequency.psd_array_welch(segmented_array, sfreq=sfreq, n_jobs=n_jobs, n_fft=1022, n_overlap=959, average='median')
        for idx, freq in enumerate(saflow.FREQS_NAMES):
            lowfreq = saflow.FREQS[idx][0]
            highfreq = saflow.FREQS[idx][1]
            freq_mask = np.where((welch_array[1] >= lowfreq) & (welch_array[1] <= highfreq))[0]
            welch_array_masked = welch_array[0][:,:,freq_mask]
            psd_array.append(np.nanmean(welch_array_masked, axis=-1))
        psd_array = np.array(psd_array)
        return psd_array, events_idx, events_dicts
    
    elif method == 'fooof':
        segmented_array, events_idx, events_dicts = segment_sourcelevel(data, filepaths, sfreq=sfreq, n_events_window=n_trials)

        # Define `peak_width_limit` setting
        peak_width = [1, 8]
        # Define `max_n_peaks` setting
        n_peaks = 4
        # Define `min_peak_height` setting
        peak_height = 0.10
        # Define frequency range
        PSD_range = [2, 120]
        # Initialize a model object for spectral parameterization, with some settings 
        fg = FOOOFGroup(peak_width_limits=peak_width, max_n_peaks=n_peaks, min_peak_height=peak_height, verbose=False, aperiodic_mode='fixed')
        bands = {key: value for key, value in zip(saflow.FREQS_NAMES, saflow.FREQS)}

        welch_array = mne.time_frequency.psd_array_welch(segmented_array, sfreq=sfreq, n_jobs=n_jobs, n_fft=1022, n_overlap=959, average='median')
        fooof_array = []
        
        psd_array = np.zeros((len(saflow.FREQS_NAMES), welch_array[0].shape[0], welch_array[0].shape[1]))
        for epoch_idx in range(welch_array[0].shape[0]):
            print(f'Computing epoch : {epoch_idx}')
            epoch_data = welch_array[0][epoch_idx,:,:]
            fg_temp = fg.copy()
            fg_temp.fit(welch_array[1], epoch_data, PSD_range, n_jobs=n_jobs)
            print(f'{fg_temp.n_null_} failed fits : {fg_temp.null_inds_}')
            
            fooof_list = []
            for chan_idx in range(welch_array[0].shape[1]):
                fm_temp = fg_temp.get_fooof(chan_idx).copy()
                fooof_list.append(fm_temp)
                fooof_freqs = fm_temp.freqs
                for band_idx, band in enumerate(saflow.FREQS_NAMES):
                    freq_mask = np.where((fooof_freqs >= bands[band][0]) & (fooof_freqs <= bands[band][1]))[0]
                    welch_array_masked = fm_temp.fooofed_spectrum_[freq_mask]
                    psd_array[band_idx, epoch_idx, chan_idx] = np.nanmean(welch_array_masked)
                del fm_temp
            del fg_temp
            fooof_list = np.array(fooof_list)
            fooof_array.append(fooof_list)
        fooof_array = np.array(fooof_array)

        return psd_array, events_idx, events_dicts, fooof_array


if __name__ == "__main__":
    args = parser.parse_args()
    n_trials = args.n_trials
    subject = args.subject
    run = args.run
    level = args.level
    method = args.method
    n_jobs = args.n_jobs
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
            filepaths['psd'].update(root=op.join('/'.join(str(filepaths['psd'].root).split('/')[:-1]), str(filepaths['psd'].root).split('/')[-1] + f'_{method}_{level}_{n_trials}trials'))

            filepaths['psd'].mkdir(exist_ok=True)
            if level == 'source':
                stc = mne.read_source_estimate(filepaths['morph'])
                data = np.float64(stc.data)
                sfreq = stc.sfreq
                
            elif level == 'sensor':
                raw = mne_bids.read_raw_bids(filepaths['preproc'])
                data = raw.get_data()
                sfreq = raw.info['sfreq']
                meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)
                data = data[meg_picks,:]

            if method == 'fooof':
                psd_array, events_idx, events_dicts, fooof_array = compute_PSD(data, filepaths, sfreq, n_trials=n_trials, method=method, n_jobs=n_jobs)
                for idx in range(psd_array.shape[1]):
                    array = psd_array[:,idx,:] # shape (n_freqs, n_channels)
                    fname = str(filepaths['psd'].fpath).replace('idx', str(events_idx[idx])) + '.pkl'
                    with open(fname, 'wb') as f:
                        pickle.dump({'data':array,
                                    'info':events_dicts[idx],}, 
                                    f)
                    with open(fname.replace('_desc-', '_fg-'), 'wb') as f:
                        pickle.dump({'fooof':fooof_array[idx,:]}, f)
            else:
                psd_array, events_idx, events_dicts = compute_PSD(data, filepaths, sfreq, n_trials=n_trials, method=method, n_jobs=n_jobs)
                for idx in range(psd_array.shape[1]):
                    array = psd_array[:,idx,:] # shape (n_freqs, n_channels)
                    fname = str(filepaths['psd'].fpath).replace('idx', str(events_idx[idx])) + '.pkl'
                    with open(fname, 'wb') as f:
                        pickle.dump({'data':array,
                                    'info':events_dicts[idx]}, 
                                    f)