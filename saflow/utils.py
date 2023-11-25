import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import saflow
from mne_bids import BIDSPath
import mne_bids
import numpy as np
import pandas as pd
import mne
import pickle

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
    
    welch_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            suffix='meg',
                            root=bids_root + '/derivatives/welch/')
    welch_bidspath.mkdir(exist_ok=True)

    lzc_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            description='idx',
                            root=bids_root + '/derivatives/lzc/')
    lzc_bidspath.mkdir(exist_ok=True)

    
    return {'raw':raw_bidspath,
            'morph':morph_bidspath,
            'preproc':preproc_bidspath,
            'psd':psd_bidspath,
            'lzc':lzc_bidspath,
            'welch':welch_bidspath,
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

    # Grab ARlog : 
    arlog_fname = str(filepaths['preproc'].copy().update(description='ARlog', processing=None).fpath)+'.pkl'
    with open(arlog_fname, 'rb') as f:
        ARlog = pickle.load(f)
    bad_epochs = ARlog.bad_epochs

    # Segment array
    segmented_array = []
    events_idx = []
    events_dict = []
    stim_events_list = []
    for idx, event in enumerate(events):
        if event[2] in [1,2]:
            stim_events_list.append(idx)
            if event[0]+tmax_samples < data_array.shape[1]:
                if event[0]+tmin_samples > 0:
                    if len(stim_events_list) >= n_events_window:
                    #if idx - n_events_window >= 0: # Check if there are enough events before the current one
                        segmented_array.append(data_array[:,event[0]+tmin_samples:event[0]+tmax_samples])
                        events_idx.append(idx)
                        # Fill a dict with events info
                        included_events = stim_events_list[-n_events_window:]
                        event_dict = {'event_idx':idx,
                                    't0_sample':event[0],
                                    'VTC':events_full.loc[idx, 'VTC'],
                                    'task':events_full.loc[idx, 'task'],
                                    'RT':events_full.loc[idx, 'RT'],
                                    'INOUT':events_full.loc[idx, 'INOUT_50_50'],
                                    'INOUT_2575':events_full.loc[idx, 'INOUT_25_75'],
                                    'INOUT_1090':events_full.loc[idx, 'INOUT_10_90'],
                                    'bad_epoch':bad_epochs[idx],
                                    'included_bad_epochs':bad_epochs[included_events],
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

def get_meg_picks_and_info(subject, run, bids_root=saflow.BIDS_PATH):
    rawpath = mne_bids.BIDSPath(subject=subject, task='gradCPT', run=run, datatype='meg', suffix='meg', root=bids_root)
    raw = mne_bids.read_raw_bids(rawpath, verbose=False)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)
    raw_mag = raw.copy().pick_types(meg=True, ref_meg=False, eeg=False, eog=False)
    return picks, raw_mag.info


def get_SAflow_bids(BIDS_PATH, subj, run, stage, cond=None):
    """
    Constructs BIDS basename and filepath in the SAflow database format.
    """
    if run == "1" or run == "8":  # determine task based on run number
        task = "RS"
    else:
        task = "gradCPT"

    if stage == "raw_ds":
        SAflow_bidsname = "sub-{}_ses-recording_task-{}_run-0{}_meg.ds".format(
            subj, task, run
        )
        SAflow_bidspath = os.path.join(
            BIDS_PATH, "sub-{}".format(subj), "ses-recording", "meg", SAflow_bidsname
        )
        return SAflow_bidsname, SAflow_bidspath
    else:
        if (
            not ("report" in stage)
            and "epo" in stage
            or "raw" in stage
            or "ENV" in stage
        ):  # determine extension based on stage
            extension = ".fif"
        elif "sources" in stage or "TFR" in stage:
            extension = ".hd5"
        elif "events" in stage:
            extension = ".tsv"
        elif "ARlog" in stage or "PSD" in stage:
            extension = ".pkl"
        elif "report" in stage:
            extension = ".html"

        if cond == None or "events" in stage:  # build basename with or without cond
            SAflow_bidsname = "sub-{}_ses-recording_task-{}_run-0{}_meg_{}{}".format(
                subj, task, run, stage, extension
            )
        else:
            SAflow_bidsname = "sub-{}_ses-recording_task-{}_run-0{}_meg_{}_{}{}".format(
                subj, task, run, stage, cond, extension
            )
        SAflow_bidspath = os.path.join(
            BIDS_PATH, "sub-{}".format(subj), "ses-recording", "meg", SAflow_bidsname
        )
        return SAflow_bidsname, SAflow_bidspath

def create_pval_mask(pvals, alpha=0.05):
    mask = np.zeros((len(pvals),), dtype="bool")
    for i, pval in enumerate(pvals):
        if pval <= alpha:
            mask[i] = True
    return mask

# Grid topoplot

def grid_topoplot(array_data, chan_info, titles_x, titles_y, masks=None, mask_params=None, cmap=None, vlims=None, title=None):
    '''Creates a grid of topoplots from the array_data. First dimension is used for the rows, second for the columns'''
    letters = ['A', 'B', 'C']
    if vlims is None:
        vlims = [(-np.max(abs(row_data)), np.max(abs(row_data))) for row_data in array_data]
    fig, axes = plt.subplots(array_data.shape[0], array_data.shape[1], figsize=(3*array_data.shape[1], 3*array_data.shape[0]))
    plt.subplots_adjust(wspace=0.1, hspace=0)
    for idx_row, row in enumerate(axes):
        for idx_col, ax in enumerate(row):
            mne.viz.plot_topomap(array_data[idx_row, idx_col], 
                                chan_info, 
                                axes=ax, 
                                show=False, 
                                cmap=cmap[idx_row] if cmap is not None else 'magma',
                                mask=masks[idx_row, idx_col] if masks is not None else None,
                                vlim=vlims[idx_row] if vlims is not None else None,
                                extrapolate="local",
                                outlines="head",
                                sphere=0.15,
                                contours=0,
            )
            if idx_row == 0:
                ax.set_title(titles_x[idx_col])
            if idx_col == 0:
                ax.set_ylabel(titles_y[idx_row], fontsize=14, rotation=0, labelpad=45)
                ax.text(
                        -0.02,
                        1.01,
                        letters[idx_row],
                        transform=ax.transAxes,
                        size=14,
                        weight="bold",
                    )
    # Add a colorbar and title. For this we need to use the figure handle.
    for row_idx in range(array_data.shape[0]):
        fig.colorbar(axes[row_idx][0].images[-1], ax=axes[row_idx], orientation='vertical', fraction=.005)

    if title is not None:
        fig.suptitle(title, y=1, fontsize=18)
    return fig, axes
