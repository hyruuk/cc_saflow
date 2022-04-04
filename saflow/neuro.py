import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from autoreject import AutoReject
from scipy.io import loadmat, savemat

# from brainpipe import feature
from saflow import BIDS_PATH, LOGS_DIR, FREQS
from mne.io import read_raw_fif, read_raw_ctf

# from hytools.meg_utils import get_ch_pos
from saflow.utils import get_SAflow_bids
from saflow.behav import find_logfile, get_VTC_from_file
import random
from matplotlib.pyplot import close
from mne.time_frequency import psd_multitaper, psd_welch
import pickle
import os.path as op
import os


def find_rawfile(subj, bloc, BIDS_PATH):
    filepath = "/sub-{}/ses-recording/meg/".format(subj)
    files = os.listdir(BIDS_PATH + filepath)
    for file in files:
        if file[-8] == bloc:
            filename = file
    return filepath, filename


def saflow_preproc(filepath, savepath, reportpath, ica=True):
    report = mne.Report(verbose=True)
    raw_data = read_raw_ctf(filepath, preload=True)
    raw_data = raw_data.apply_gradient_compensation(
        grade=3
    )  # required for source reconstruction
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude="bads")
    fig = raw_data.plot(show=False)
    report.add_figure(fig, title="Time series")
    close(fig)
    fig = raw_data.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD")
    close(fig)

    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.5
    raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
    raw_data.notch_filter(
        np.arange(60, high_cutoff + 1, 60),
        picks=picks,
        filter_length="auto",
        phase="zero",
        fir_design="firwin",
    )
    fig = raw_data.plot_psd(average=False, picks=picks, fmax=120, show=False)
    report.add_figure(fig, title="PSD filtered")
    close(fig)
    if ica == False:
        report.save(reportpath, open_browser=False, overwrite=True)
        raw_data.save(savepath, overwrite=True)
        del report
        del raw_data
        del fig

    elif ica == True:
        ## ICA
        ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
        fig = ica.plot_sources(raw_data, show=False)
        report.add_figure(fig, title="Independent Components")
        close(fig)

        ## FIND ECG COMPONENTS
        ecg_threshold = 0.50
        ecg_epochs = create_ecg_epochs(raw_data, ch_name="EEG059")
        ecg_inds, ecg_scores = ica.find_bads_ecg(
            ecg_epochs, ch_name="EEG059", method="ctps", threshold=ecg_threshold
        )
        fig = ica.plot_scores(ecg_scores, ecg_inds, show=False)
        report.add_figure(fig, title="Correlation with ECG (EEG059)")
        close(fig)
        fig = list()
        try:
            fig = ica.plot_properties(
                ecg_epochs, picks=ecg_inds, image_args={"sigma": 1.0}, show=False
            )
            for i, figure in enumerate(fig):
                report.add_figure(figure, title="Detected component " + str(i))
                close(figure)
        except:
            print("No component to remove")

        ## FIND EOG COMPONENTS
        eog_threshold = 4
        eog_epochs = create_eog_epochs(raw_data, ch_name="EEG057")
        eog_inds, eog_scores = ica.find_bads_eog(
            eog_epochs, ch_name="EEG057", threshold=eog_threshold
        )
        # TODO : if eog_inds == [] then eog_inds = [index(max(abs(eog_scores)))]
        fig = ica.plot_scores(eog_scores, eog_inds, show=False)
        report.add_figure(fig, title="Correlation with EOG (EEG057)")
        close(fig)
        fig = list()
        try:
            fig = ica.plot_properties(
                eog_epochs, picks=eog_inds, image_args={"sigma": 1.0}, show=False
            )
            for i, figure in enumerate(fig):
                report.add_figure(figure, title="Detected component " + str(i))
                close(figure)
        except:
            print("No component to remove")

        ## EXCLUDE COMPONENTS
        ica.exclude = ecg_inds
        ica.apply(raw_data)
        ica.exclude = eog_inds
        ica.apply(raw_data)
        fig = raw_data.plot(show=False)
        # Plot the clean signal.
        report.add_figure(fig, title="After filtering + ICA")
        close(fig)
        ## SAVE PREPROCESSED FILE
        report.save(reportpath, open_browser=False, overwrite=True)
        raw_data.save(savepath, overwrite=True)
        del ica
        del report
        del raw_data
        del fig


def segment_files(bids_filepath, tmin=0, tmax=0.8):
    raw = read_raw_fif(bids_filepath, preload=True)
    picks = mne.pick_types(
        raw.info, meg=True, ref_meg=True, eeg=False, eog=False, stim=False
    )
    ### Set some constants for epoching
    baseline = None  # (None, -0.05)
    # reject = {'mag': 4e-12}
    try:
        events = mne.find_events(raw, min_duration=1 / raw.info["sfreq"], verbose=False)
    except ValueError:
        events = mne.find_events(raw, min_duration=2 / raw.info["sfreq"], verbose=False)
    event_id = {"Freq": 21, "Rare": 31}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=None,
        picks=picks,
        preload=True,
    )
    ar = AutoReject(n_jobs=24)
    epochs_clean, autoreject_log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean, autoreject_log


def remove_errors(logfile, events):
    """
    Takes as input the raw events, including both stimuli and responses.
    Outputs an event vector containing only the correct trials, and indices of their position in the list
    of all stims
    TODO : base this on behav data instead of MNE events
    /!\ that function looks fishy <- please just tell me why -_-
    """

    # load RT vector
    data = loadmat(logfile)
    df_response = pd.DataFrame(data["response"])
    RT_array = np.asarray(df_response.loc[:, 4])

    # get events vector with only stimuli events
    new_events = []
    for ev in events:
        if ev[2] != 99:
            new_events.append(ev)
    events = np.array(new_events)

    # check for each trial if it has a response or not
    events_noerr = []

    events_comerr = []
    events_omerr = []
    events_comcorr = []
    events_omcorr = []

    for idx, event in enumerate(events):
        if event[2] == 21:
            if RT_array[idx] != 0:
                events_noerr.append(event)
                events_comcorr.append(event)
            else:
                events_omerr.append(event)
        if event[2] == 31:
            if RT_array[idx] == 0:
                events_noerr.append(event)
                events_omcorr.append(event)
            else:
                events_comerr.append(event)
    events_noerr = np.array(events_noerr)
    events_comerr = np.array(events_comerr)
    events_omerr = np.array(events_omerr)
    events_comcorr = np.array(events_comcorr)
    events_omcorr = np.array(events_omcorr)

    return events_noerr, events_comerr, events_omerr, events_comcorr, events_omcorr

def annotate_events(logfile, events):
    data = loadmat(logfile)
    df_response = pd.DataFrame(data["response"])

    annotated_events = []
    events_full = events.copy()
    idx_stim = 0
    for idx_ev, ev in enumerate(events_full):
        current_ev = ev.copy()
        if ev[2] == 21:
            if df_response.loc[idx_stim, 1] != 0.0:
                ev[2] = 211
            else:
                ev[2] = 210
            annotated_events.append(ev)
            idx_stim += 1
        elif ev[2] == 31:
            if df_response.loc[idx_stim, 1] == 0.0:
                ev[2] = 311
            else:
                ev[2] = 310
            annotated_events.append(ev)
            idx_stim += 1
    return np.array(annotated_events)


def trim_events(events_noerr, events_artrej):
    """
    This function compares the events vectors of correct epochs (events_noerr)
    and of kept epochs after auto-reject (events_artrej).
    Returns a list of intersecting epochs, + their idx in the clean epochs vector
    """
    events_trimmed = []
    idx_trimmed = []
    for idx, event in enumerate(events_artrej):
        if event[0] in events_noerr[:, 0]:
            events_trimmed.append(event)
            idx_trimmed.append(idx)

    events_trimmed = np.array(events_trimmed)
    idx_trimmed = np.array(idx_trimmed)

    print("N events in clean epochs : {}".format(len(events_artrej)))
    print("N events in correct epochs : {}".format(len(events_noerr)))
    print("N events in intersection : {}".format(len(idx_trimmed)))
    return events_trimmed, idx_trimmed


def trim_INOUT_idx(INidx, OUTidx, events_trimmed, events):
    """
    With INidx_trimmed refering to indices in events_artrej
    """
    # get events vector with only stimuli events
    new_events = []
    for ev in events:
        if ev[2] != 99:
            new_events.append(ev)
    events = np.array(new_events)

    INidx_trimmed = []
    OUTidx_trimmed = []
    # compare trimmed events with all_events, and store corresponding indices
    for idx, ev in enumerate(events):
        for idx_trim, ev_trim in enumerate(events_trimmed):
            if ev[0] == ev_trim[0]:
                if idx in INidx:
                    INidx_trimmed.append(idx_trim)
                if idx in OUTidx:
                    OUTidx_trimmed.append(idx_trim)
    INidx_trimmed = np.array(INidx_trimmed)
    OUTidx_trimmed = np.array(OUTidx_trimmed)

    return INidx_trimmed, OUTidx_trimmed


def get_odd_epochs(BIDS_PATH, LOGS_DIR, subj, bloc, stage="-epo"):
    """
    Returns an array of indices of Freqs and Rares epochs. Retains only clean epochs.
    """
    ### Get events after artifact rejection have been performed
    epo_path, epo_filename = get_SAflow_bids(
        BIDS_PATH, subj, bloc, stage=stage, cond=None
    )
    events_artrej = mne.read_events(
        epo_filename, verbose=False
    )  # get events from the epochs file (so no resp event)

    ### Get original events from the raw file, to compare them to the events left in the epochs file
    events_fname, events_fpath = get_SAflow_bids(
        BIDS_PATH, subj, bloc, stage="preproc_raw", cond=None
    )
    raw = read_raw_fif(
        events_fpath, preload=False, verbose=False
    )  # , min_duration=2/epochs.info['sfreq'])
    try:
        events = mne.find_events(raw, min_duration=1 / raw.info["sfreq"], verbose=False)
    except ValueError:
        events = mne.find_events(raw, min_duration=2 / raw.info["sfreq"], verbose=False)

    # Get the list of hits/miss events
    log_file = LOGS_DIR + find_logfile(subj, bloc, os.listdir(LOGS_DIR))
    events_noerr, events_comerr, events_omerr, events_comcorr, events_omcorr = remove_errors(log_file, events)

    # Keep only events that are clean, and split them by condition
    # Start with correct events
    events_noerr_trimmed, idx_noerr_trimmed = trim_events(events_noerr, events_artrej)
    freqs_hits_idx = np.array(
        [idx_noerr_trimmed[i] for i, x in enumerate(events_noerr_trimmed) if x[2] == 21]
    )
    rares_hits_idx = np.array(
        [idx_noerr_trimmed[i] for i, x in enumerate(events_noerr_trimmed) if x[2] == 31]
    )

    # Then commission errors
    if events_comerr.size > 0:
        events_comerr_trimmed, idx_comerr_trimmed = trim_events(
            events_comerr, events_artrej
        )
        rares_miss_idx = np.array(idx_comerr_trimmed)
    else:
        rares_miss_idx = np.array([])
    # And finally ommission errors
    if events_omerr.size > 0:
        events_omerr_trimmed, idx_omerr_trimmed = trim_events(
            events_omerr, events_artrej
        )
        freqs_miss_idx = np.array(idx_omerr_trimmed)
    else:
        freqs_miss_idx = np.array([])

    return freqs_hits_idx, freqs_miss_idx, rares_hits_idx, rares_miss_idx


def get_VTC_epochs(
    BIDS_PATH,
    LOGS_DIR,
    subj,
    run,
    stage="-epo",
    lobound=None,
    hibound=None,
    save_epochs=False,
    filt_order=3,
    filt_cutoff=0.1,
):
    """
    This functions allows to use the logfile to split the epochs obtained in the epo.fif file.
    It works by comparing the timestamps of IN and OUT events to the timestamps in the epo file events
    It returns IN and OUT indices that are to be used in the split_PSD_data function

    """
    ### Get events after artifact rejection have been performed
    epo_path, epo_filename = get_SAflow_bids(
        BIDS_PATH, subj, run, stage=stage, cond=None
    )
    events_artrej = mne.read_events(
        epo_filename, verbose=False
    )  # get events from the epochs file (so no resp event)

    ### Find logfile to extract VTC
    behav_list = os.listdir(LOGS_DIR)
    log_file = LOGS_DIR + find_logfile(subj, run, behav_list)

    (
        INidx,
        OUTidx,
        VTC_raw,
        VTC_filtered,
        IN_mask,
        OUT_mask,
        performance_dict,
        df_response_out,
    ) = get_VTC_from_file(
        subj, run, behav_list, inout_bounds=[lobound, hibound], filt_cutoff=filt_cutoff
    )

    ### Get original events and split them using the VTC
    events_fname, events_fpath = get_SAflow_bids(
        BIDS_PATH, subj, run, stage="preproc_raw", cond=None
    )
    raw = read_raw_fif(
        events_fpath, preload=False, verbose=False
    )  # , min_duration=2/epochs.info['sfreq'])
    try:
        events = mne.find_events(raw, min_duration=1 / raw.info["sfreq"], verbose=False)
    except ValueError:
        events = mne.find_events(raw, min_duration=2 / raw.info["sfreq"], verbose=False)

    events_noerr, events_comerr, events_omerr, events_comcorr, events_omcorr = remove_errors(log_file, events)
    # Keep only events that are correct and clean
    events_trimmed, idx_trimmed = trim_events(events_comcorr, events_artrej)
    # Write INidx and OUTidx as indices of clean events
    INidx, OUTidx = trim_INOUT_idx(INidx, OUTidx, events_trimmed, events)

    VTC_epo = np.array([VTC_raw[idx] for idx in idx_trimmed])

    return INidx, OUTidx, VTC_epo, idx_trimmed


def compute_PSD(epochs, freqlist=FREQS, method="multitaper", tmin=0, tmax=0.8):
    epochs_psds = []

    # Compute PSD
    if method == "multitaper":
        psds, freqs = psd_multitaper(
            epochs, fmin=min(min(freqlist)), fmax=max(max(freqlist)), n_jobs=1
        )
    if method == "pwelch":
        psds, freqs = psd_welch(
            epochs,
            average="median",
            fmin=min(min(freqlist)),
            fmax=max(max(freqlist)),
            n_jobs=1,
        )
    if method == "pwelch" or method == "multitaper":
        psds = 10.0 * np.log10(psds)  # Convert power to dB scale.
        # Average in freq bands
        for low, high in freqlist:
            freq_idx = [i for i, x in enumerate(freqs) if x >= low and x <= high]
            psd = np.mean(psds[:, :, freq_idx], axis=2)
            epochs_psds.append(psd)
        epochs_psds = np.array(epochs_psds).swapaxes(2, 0).swapaxes(1, 0)

        # TODO : compute via hilbert
    if method == "hilbert":
        for low, high in freqlist:
            # Filter continuous data
            data = epochs.copy().filter(low, high)  # Here epochs is a raw file
            hilbert = data.apply_hilbert(envelope=True)
            hilbert_pow = hilbert.copy()
            hilbert_pow._data = hilbert._data**2

            # Segment them
            picks = mne.pick_types(
                epochs.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
            )
            try:
                events = mne.find_events(
                    epochs, min_duration=1 / epochs.info["sfreq"], verbose=False
                )
            except ValueError:
                events = mne.find_events(
                    epochs, min_duration=2 / epochs.info["sfreq"], verbose=False
                )
            event_id = {"Freq": 21, "Rare": 31}
            epochs = mne.Epochs(
                hilbert_pow,
                events=events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                reject=None,
                picks=picks,
                preload=True,
            )
            epochs.drop(ARlog.bad_epochs)
            epochs_psds.append(epochs.get_data())
        epochs_psds = np.array(epochs_psds)
        epochs_psds = np.mean(epochs_psds, axis=3).transpose(1, 2, 0)
        print(epochs_psds.shape)
    return epochs_psds


def compute_PSD_hilbert(raw, ARlog, tmin=0, tmax=0.8, freqlist=FREQS):
    epochs_psds = []
    for low, high in freqlist:
        # Filter continuous data
        data = raw.copy().filter(low, high)  # Here epochs is a raw file
        hilbert = data.apply_hilbert(envelope=True)
        hilbert_pow = hilbert.copy()
        hilbert_pow._data = hilbert._data**2

        # Segment them
        picks = mne.pick_types(
            raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
        )
        try:
            events = mne.find_events(
                raw, min_duration=1 / raw.info["sfreq"], verbose=False
            )
        except ValueError:
            events = mne.find_events(
                raw, min_duration=2 / raw.info["sfreq"], verbose=False
            )
        event_id = {"Freq": 21, "Rare": 31}
        epochs = mne.Epochs(
            hilbert,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            picks=picks,
            preload=True,
        )
        epochs.drop(ARlog.bad_epochs)
        epochs_psds.append(epochs.get_data())
    epochs_psds = np.array(epochs_psds)
    epochs_psds = np.mean(epochs_psds, axis=3).transpose(1, 2, 0)
    print(epochs_psds.shape)
    return epochs_psds


def compute_envelopes_hilbert(raw, ARlog, freqlist=None, tmin=0, tmax=0.8):
    epochs_envelopes = []
    if freqlist == None:
        freqlist = [[4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120]]
    for low, high in freqlist:
        # Filter continuous data
        data = raw.copy().filter(low, high)
        # Segment them
        picks = mne.pick_types(
            raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
        )
        try:
            events = mne.find_events(
                raw, min_duration=1 / raw.info["sfreq"], verbose=False
            )
        except ValueError:
            events = mne.find_events(
                raw, min_duration=2 / raw.info["sfreq"], verbose=False
            )
        event_id = {"Freq": 21, "Rare": 31}

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            picks=picks,
            preload=True,
        )
        # Drop bad epochs
        epochs.drop(ARlog.bad_epochs)
        # Apply Hilbert transform to obtain envelope
        hilbert = epochs.apply_hilbert(envelope=True)
        hilbert_pow = hilbert.copy()
        hilbert_pow._data = hilbert._data**2
        epochs_envelopes.append(hilbert_pow)
        del hilbert_pow
        del hilbert
        del data
        del epochs
    return epochs_envelopes


def compute_TFR(epochs, baseline=False):
    decim = 3
    fmin = 2
    fmax = 150
    n_bins = 30
    freqs = np.logspace(*np.log10([fmin, fmax]), num=n_bins)
    # freqs = np.arange(2,15,1)  # define frequencies of interest
    n_cycles = freqs / freqs[0]
    zero_mean = False
    this_tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        average=False,
        zero_mean=zero_mean,
        return_itc=False,
    )
    if baseline:
        this_tfr.apply_baseline(mode="ratio", baseline=(None, -0.05))
    # this_power = this_tfr.data[:, :, :, :]  # we only have one channel.
    return this_tfr


def load_PSD_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, time_avg=True, stage="PSD"):
    """
    Returns a list containing n_subj lists of n_blocs matrices of shape n_freqs X n_channels X n_trials
    """
    PSD_alldata = []
    for subj in SUBJ_LIST:
        all_subj = []  ## all the data of one subject
        for run in BLOCS_LIST:
            SAflow_bidsname, SAflow_bidspath = get_SAflow_bids(
                BIDS_PATH, subj, run, stage=stage, cond=None
            )
            mat = loadmat(SAflow_bidspath)["PSD"]
            if time_avg == True:
                mat = np.mean(mat, axis=2)  # average PSDs in time across epochs
            all_subj.append(mat)
        PSD_alldata.append(all_subj)
    return PSD_alldata


def load_VTC_data(BIDS_PATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST):
    VTC_alldata = []
    for subj in SUBJ_LIST:
        all_subj = []  ## all the data of one subject
        for run in BLOCS_LIST:
            # get events from epochs file
            epo_path, epo_filename = get_SAflow_bids(
                BIDS_PATH, subj, run, "epo", cond=None
            )
            events_epoched = mne.read_events(
                epo_filename, verbose=False
            )  # get events from the epochs file (so no resp event)
            # get events from original file (only 599 events)
            events_fname, events_fpath = get_SAflow_bids(
                BIDS_PATH, subj, run, stage="preproc_raw", cond=None
            )
            raw = read_raw_fif(
                events_fpath, preload=False, verbose=False
            )  # , min_duration=2/epochs.info['sfreq'])
            all_events = mne.find_events(
                raw, min_duration=2 / raw.info["sfreq"], verbose=False
            )
            stim_idx = []
            for i in range(len(all_events)):
                if all_events[i, 2] in [21, 31]:
                    stim_idx.append(i)
            all_events = all_events[stim_idx]
            # compute VTC for all trials
            log_file = find_logfile(subj, run, os.listdir(LOGS_DIR))
            VTC, INbounds, OUTbounds, INzone, OUTzone = get_VTC_from_file(
                LOGS_DIR + log_file, lobound=None, hibound=None
            )
            epochs_VTC = []
            for event_time in events_epoched[:, 0]:
                idx = list(all_events[:, 0]).index(event_time)
                epochs_VTC.append(VTC[idx])
            all_subj.append(np.array(epochs_VTC))
        VTC_alldata.append(all_subj)
    return VTC_alldata


def split_TFR(
    BIDS_PATH,
    subj,
    bloc,
    by="VTC",
    lobound=None,
    hibound=None,
    stage="1600TFR",
    filt_order=3,
    filt_cutoff=0.05,
):
    if by == "VTC":
        event_id = {"IN": 1, "OUT": 0}
        INidx, OUTidx, VTC_epochs, idx_trimmed = get_VTC_epochs(
            LOGS_DIR,
            subj,
            bloc,
            lobound=lobound,
            hibound=hibound,
            stage=stage[:-3] + "epo",
            save_epochs=False,
            filt_order=filt_order,
            filt_cutoff=filt_cutoff,
        )
        epo_path, epo_filename = get_SAflow_bids(
            BIDS_PATH, subj, bloc, stage=stage[:-3] + "epo", cond=None
        )
        epo_events = mne.read_events(
            epo_filename, verbose=False
        )  # get events from the epochs file (so no resp event)
        TFR_path, TFR_filename = get_SAflow_bids(
            BIDS_PATH, subj, bloc, stage=stage, cond=None
        )
        TFR = mne.time_frequency.read_tfrs(TFR_filename)
        for i, event in enumerate(epo_events):
            if i in INidx:
                TFR[0].events[i, 2] = 1
            if i in OUTidx:
                TFR[0].events[i, 2] = 0
        TFR[0].event_id = event_id
    return TFR


def split_PSD_data(
    BIDS_PATH,
    SUBJ_LIST,
    BLOCS_LIST,
    by="VTC",
    lobound=None,
    hibound=None,
    stage="PSD",
    filt_order=3,
    filt_cutoff=0.1,
):
    """
    This func splits the PSD data into two conditions. It returns a list of 2 (cond1 and cond2), each containing a list of n_subject matrices of shape n_freqs X n_channels X n_trials
    """
    PSD_alldata = load_PSD_data(
        BIDS_PATH, SUBJ_LIST, BLOCS_LIST, time_avg=True, stage=stage
    )
    PSD_cond1 = []
    PSD_cond2 = []
    for subj_idx, subj in enumerate(SUBJ_LIST):
        subj_cond1 = []
        subj_cond2 = []
        for bloc_idx, bloc in enumerate(BLOCS_LIST):
            print("Splitting sub-{}_run-{}".format(subj, bloc))

            # Obtain indices of the two conditions
            if by == "VTC":
                INidx, OUTidx, VTC_epochs, idx_trimmed = get_VTC_epochs(
                    LOGS_DIR,
                    subj,
                    bloc,
                    lobound=lobound,
                    hibound=hibound,
                    save_epochs=False,
                    filt_order=filt_order,
                    filt_cutoff=filt_cutoff,
                )
                cond1_idx = INidx
                cond2_idx = OUTidx
            if by == "odd":
                # Get indices of freq and rare events
                ev_fname, ev_fpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage="epo")
                events_artrej = mne.read_events(ev_fpath)
                log_file = LOGS_DIR + find_logfile(subj, bloc, os.listdir(LOGS_DIR))
                events_fname, events_fpath = get_SAflow_bids(
                    BIDS_PATH, subj, bloc, stage="preproc_raw", cond=None
                )
                raw = read_raw_fif(
                    events_fpath, preload=False, verbose=False
                )  # , min_duration=2/epochs.info['sfreq'])
                try:
                    events = mne.find_events(
                        raw, min_duration=1 / raw.info["sfreq"], verbose=False
                    )
                except ValueError:
                    events = mne.find_events(
                        raw, min_duration=2 / raw.info["sfreq"], verbose=False
                    )

                events_noerr, events_comerr, events_omerr = remove_errors(
                    log_file, events
                )
                events_trimmed, idx_trimmed = trim_events(events_noerr, events_artrej)
                cond1_idx = []
                cond2_idx = []
                for idx, ev in enumerate(events_trimmed):
                    if ev[2] == 21:  # Frequent events
                        cond1_idx.append(idx)
                    if ev[2] == 31:
                        cond2_idx.append(idx)
                cond1_idx = np.array(cond1_idx)
                cond2_idx = np.array(cond2_idx)
                # Add this to keep the same number of trials in both conditions
                random.seed(0)
                cond1_idx = random.choices(cond1_idx, k=len(cond2_idx))
                print(
                    "N trials retained for each condition : {}".format(len(cond2_idx))
                )
            # Pick the data of each condition
            if bloc_idx == 0:  # if first bloc, init ndarray size using the first matrix
                subj_cond1 = PSD_alldata[subj_idx][bloc_idx][:, :, cond1_idx]
                subj_cond2 = PSD_alldata[subj_idx][bloc_idx][:, :, cond2_idx]
            else:  # if not first bloc, just concatenate along the trials dimension
                subj_cond1 = np.concatenate(
                    (subj_cond1, PSD_alldata[subj_idx][bloc_idx][:, :, cond1_idx]),
                    axis=2,
                )
                subj_cond2 = np.concatenate(
                    (subj_cond2, PSD_alldata[subj_idx][bloc_idx][:, :, cond2_idx]),
                    axis=2,
                )
        PSD_cond1.append(subj_cond1)
        PSD_cond2.append(subj_cond2)
    splitted_PSD = [PSD_cond1, PSD_cond2]
    return splitted_PSD


def split_trials(
    BIDS_PATH,
    LOGS_DIR,
    subj,
    run,
    stage="PSD",
    by="VTC",
    keep_errors=False,
    equalize=False,
    lobound=None,
    hibound=None,
    filt_order=3,
    filt_cutoff=0.05,
    freq_names=None,
    oddball="hits",
):
    """
    If stage is 'env', freq_names must be specified.
    """
    # Split epochs indices
    if by == "VTC":
        INidx, OUTidx, VTC_epochs, idx_trimmed = get_VTC_epochs(
            BIDS_PATH,
            LOGS_DIR,
            subj,
            run,
            lobound=lobound,
            hibound=hibound,
            save_epochs=False,
            filt_order=filt_order,
            filt_cutoff=filt_cutoff,
        )
        condA_idx = INidx
        condB_idx = OUTidx
        print("{} IN epochs".format(len(INidx)))
        print("{} OUT epochs".format(len(OUTidx)))
    elif by == "odd":
        freqs_hits_idx, freqs_miss_idx, rares_hits_idx, rares_miss_idx = get_odd_epochs(
            BIDS_PATH, LOGS_DIR, subj, run, stage="-epo"
        )
        if oddball == "all":
            condA_idx = np.sort(np.concatenate((freqs_hits_idx, freqs_miss_idx)))
            condB_idx = np.sort(np.concatenate((rares_hits_idx, rares_miss_idx)))
        elif oddball == "hits":
            condA_idx = freqs_hits_idx
            condB_idx = rares_hits_idx
        elif oddball == "miss":
            condA_idx = freqs_miss_idx
            condB_idx = rares_miss_idx
        elif oddball == "rares":
            condA_idx = rares_hits_idx
            condB_idx = rares_miss_idx
        print("{} condA epochs".format(len(condA_idx)))
        print("{} condB epochs".format(len(condB_idx)))
    elif by == "resp":
        freqs_hits_idx, freqs_miss_idx, rares_hits_idx, rares_miss_idx = get_odd_epochs(
            BIDS_PATH, LOGS_DIR, subj, run, stage="-epo"
        )
        condA_idx = np.sort(np.concatenate((freqs_hits_idx, rares_miss_idx)))
        condB_idx = np.sort(np.concatenate((freqs_miss_idx, rares_hits_idx)))
        print("{} Resp epochs".format(len(condA_idx)))
        print("{} NoResp epochs".format(len(condB_idx)))
        condA_idx = [int(x) for x in condA_idx]
        condB_idx = [int(x) for x in condB_idx]
    # Load epochs
    if stage == "PSD":
        fname, fpath = get_SAflow_bids(BIDS_PATH, subj, run, stage, cond=None)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        try:
            condA = data[condA_idx]
            condB = data[condB_idx]
        except IndexError:
            print("condB empty")
            condB = []
    elif "env" in stage:
        condA = []
        condB = []
        for freq in freq_names:
            fname, fpath = get_SAflow_bids(BIDS_PATH, subj, run, stage, cond=freq)
            data = mne.read_epochs(fpath)
            condA.append(data[condA_idx])
            condB.append(data[condB_idx])
    return condA, condB
