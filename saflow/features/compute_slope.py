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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='23',
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

def create_fnames(subject, run, bids_root=saflow.BIDS_PATH):
    morph_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            processing='clean',
                            description='morphed',
                            root=bids_root + '/derivatives/morphed_sources/')
    
    preproc_bidspath = BIDSPath(subject=subject, 
                        task='gradCPT', 
                        run=run, 
                        datatype='meg', 
                        suffix='meg',
                        processing='clean',
                        root=bids_root + '/derivatives/preprocessed/')
    
    psd_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            description='psd_idx',
                            root=bids_root + '/derivatives/psd/')
    psd_bidspath.mkdir(exist_ok=True)

    lzc_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            description='lzc_idx',
                            root=bids_root + '/derivatives/lzc/')
    lzc_bidspath.mkdir(exist_ok=True)
    
    return {'morph':morph_bidspath,
            'preproc':preproc_bidspath,
            }


def segment_sourcelevel(data_array, filepaths, sfreq=600, tmin=0.426, tmax=1.278):
    """
    Segment a source-level data array based on events from a preprocessed MEG dataset.

    Parameters
    ----------
    data_array : np.ndarray
        The source-level data array to segment.
    preproc_bidspath : mne_bids.BIDSPath
        The BIDSPath object pointing to the preprocessed MEG dataset.
    sfreq : float, optional
        The sampling frequency of the data, in Hz. Default is 600.
    tmin : float, optional
        The start time of the segment relative to the event onset, in seconds. Default is 0.426.
    tmax : float, optional
        The end time of the segment relative to the event onset, in seconds. Default is 1.278.

    Returns
    -------
    segmented_array : np.ndarray
        The segmented data array, with shape (n_events, n_channels, n_samples).
    """
    # Load events
    preproc = mne_bids.read_raw_bids(bids_path=filepaths['preproc'], verbose=False)
    events = mne.events_from_annotations(preproc, verbose=False)
    # Compute time samples
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)

    # Segment array
    segmented_array = []
    events_idx = []
    for idx, event in enumerate(events[0]):
        if event[2] in [1,2]:
            if event[0]+tmax_samples < data_array.shape[1]:
                segmented_array.append(data_array[:,event[0]+tmin_samples:event[0]+tmax_samples])
                events_idx.append(idx)
    segmented_array = np.array(segmented_array)
    events_idx = np.array(events_idx)
    return segmented_array, events_idx

def compute_slope(stc, filepaths, fooof_bounds=[5,40]):
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
    segmented_array, events_idx = segment_sourcelevel(stc.data, filepaths['preproc'], sfreq=stc.sfreq)
    slope_array = []
    for idx, epoch in enumerate(segmented_array):
        epoch_array = []
        for channel in epoch:
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
        slope_array.append(epoch_array)
        
        # Save epoch
        fname = filepaths['lzc'].replace('_idx', '_'+str(events_idx[idx])) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(epoch_array, f)

    slope_array = np.array(slope_array)
    return slope_array, events_idx

if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    run = args.run
    freq_bands = saflow.FREQS
    freq_names = saflow.FREQS_NAMES

    filepaths = create_fnames(subject, run)

    stc = mne.read_source_estimate(filepaths['morph'])

    slope_array, events_idx = compute_slope(stc, filepaths)
