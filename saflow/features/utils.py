import saflow
from mne_bids import BIDSPath
import mne_bids
import numpy as np
import pandas as pd
import mne

def create_fnames(subject, run, bids_root=saflow.BIDS_PATH):
    morph_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            processing='clean',
                            description='morphed',
                            root=bids_root + '/derivatives/morphed_sources/')
    raw_bidspath = BIDSPath(subject=subject,
                    task='gradCPT',
                    run=run,
                    datatype='meg',
                    suffix='meg',
                    root=bids_root)
    
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
                            description='idx',
                            root=bids_root + '/derivatives/psd/')
    psd_bidspath.mkdir(exist_ok=True)

    lzc_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            description='idx',
                            root=bids_root + '/derivatives/lzc/')
    lzc_bidspath.mkdir(exist_ok=True)

    slope_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            description='idx',
                            root=bids_root + '/derivatives/FOOOF_slope/')
    slope_bidspath.mkdir(exist_ok=True)
    
    return {'raw':raw_bidspath,
            'morph':morph_bidspath,
            'preproc':preproc_bidspath,
            'psd':psd_bidspath,
            'lzc':lzc_bidspath,
            'slope':slope_bidspath,
            }


def segment_sourcelevel(data_array, filepaths, sfreq=600, tmin=0.426, tmax=1.278, n_events_window=1):
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
    events, event_id = mne.events_from_annotations(preproc, verbose=False)
    events_full = pd.read_csv(str(filepaths['raw'].fpath).replace('_meg.ds', '_events.tsv'), sep='\t')

    # Compute time samples
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)
    epoch_length = tmax_samples - tmin_samples
    tmin_samples = tmin_samples - int(epoch_length * (n_events_window-1))


    # Segment array
    segmented_array = []
    events_idx = []
    events_dict = []
    for idx, event in enumerate(events):
        if event[2] in [1,2]:
            if event[0]+tmax_samples < data_array.shape[1]:
                if event[0]+tmin_samples > 0:
                    segmented_array.append(data_array[:,event[0]+tmin_samples:event[0]+tmax_samples])
                    events_idx.append(idx)

                    # Fill a dict with events info
                    included_events = [idx - i for i in range(n_events_window)]
                    event_dict = {'event_idx':idx,
                                  't0_sample':event[0],
                                  'VTC':events_full.loc[idx, 'VTC'],
                                  'task':events_full.loc[idx, 'task'],
                                  'RT':events_full.loc[idx, 'RT'],
                                  'INOUT':events_full.loc[idx, 'INOUT_50_50'],
                                  'INOUT_2575':events_full.loc[idx, 'INOUT_25_75'],
                                  'INOUT_1090':events_full.loc[idx, 'INOUT_10_90'],
                                  'included_events_idx':included_events,
                                  'included_VTC':events_full.loc[included_events, 'VTC'].values,
                                  'included_task':events_full.loc[included_events, 'task'].values,
                                  'included_RT':events_full.loc[included_events, 'RT'].values,
                                  'included_INOUT':events_full.loc[included_events, 'INOUT_50_50'].values,
                                  'included_INOUT_2575':events_full.loc[included_events, 'INOUT_25_75'].values,
                                  'included_INOUT_1090':events_full.loc[included_events, 'INOUT_10_90'].values,
                                  }
                    events_dict.append(event_dict)
    segmented_array = np.array(segmented_array)
    events_idx = np.array(events_idx)
    return segmented_array, events_idx, events_dict