import os
import os.path as op
import pickle

from saflow import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import argparse
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from autoreject import AutoReject
import mne
from mne.io import read_raw_ctf
from matplotlib.pyplot import close
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='05',
    type=str,
    help="Subject to process",
)

def create_fnames(subject, bloc, bids_root = BIDS_PATH):
    # Setup input files
    raw_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            suffix='meg',
                            extension='.ds',
                            root=bids_root)
    
    preproc_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            suffix='meg',
                            processing='clean',
                            root=bids_root + '/derivatives/preprocessed/')
    
    ARlog_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg',
                            suffix='meg',
                            description='ARlog',
                            root=bids_root + '/derivatives/preprocessed/')
    
    report_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg',
                            suffix='meg',
                            description='report',
                            root=bids_root + '/derivatives/preprocessed/')

    epoch_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            processing='epo',
                            root=bids_root + '/derivatives/epochs/')
    
    return {'raw':raw_bidspath,
            'preproc':preproc_bidspath,
            'epoch':epoch_bidspath,
            'ARlog':ARlog_bidspath,
            'report':report_bidspath}

def load_noise_cov(er_date, bids_root=BIDS_PATH):
    # Noise covariance matrix
    noise_cov_bidspath = BIDSPath(subject='emptyroom', 
                    session=er_date,
                    task='noise', 
                    datatype="meg",
                    processing='noisecov',
                    root=bids_root + '/derivatives/noise_cov/')
    noise_cov_fullpath = str(noise_cov_bidspath.fpath)
    if not os.path.isfile(noise_cov_fullpath):
        er_raw = read_raw_bids(BIDSPath(subject='emptyroom', 
                    session=er_date,
                    task='noise', 
                    datatype="meg",
                    root=bids_root))
        os.makedirs(os.path.dirname(noise_cov_bidspath.fpath), exist_ok=True)
        noise_cov = mne.compute_raw_covariance(
                    er_raw, method=["shrunk", "empirical"], rank=None, verbose=True
                )
        noise_cov.save(noise_cov_fullpath)
    else:
        noise_cov = mne.read_cov(noise_cov_fullpath)
    return noise_cov

def preproc_pipeline(filepaths, tmin, tmax):
    # Init report
    report = mne.Report(verbose=True)
    # Load raw data
    raw = read_raw_bids(filepaths['raw'], {'preload':True})
    events, event_id = mne.events_from_annotations(raw)
    raw, events = raw.resample(600, events=events)
    picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True)

    raw = raw.apply_gradient_compensation(grade=3)  # required for source reconstruction
    
    # Plot raw signal
    report.add_raw(raw, title="Raw data")
    #report.add_events(mne.events_from_annotations(raw)[0], title="Events")
    fig = raw.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series")
    fig = raw.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD")


    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.1
    line_freq = raw.info['line_freq']
    preproc = raw.copy().filter(low_cutoff, high_cutoff, picks=picks, fir_design="firwin")
    preproc.notch_filter(
        np.arange(line_freq, high_cutoff + 1, line_freq),
        picks=picks,
        filter_length="auto",
        phase="zero",
        fir_design="firwin",
    )
    # Create filtered copy for Autoreject and ICA fitting
    raw_filt = preproc.copy().filter(1, None)

    ## Plot filtered signal
    report.add_raw(preproc, title="Filtered data")
    fig = preproc.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series (filtered)")
    fig = preproc.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (filtered)")
    
    report.add_raw(raw_filt, title="Filtered data (for AR)")
    fig = raw_filt.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series (filtered for AR)")
    fig = raw_filt.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (filtered for AR)")

    ## Epoching for Autoreject
    epochs_filt = mne.Epochs(
        raw_filt,
        events=events,
        event_id=event_id,
        tmin=tmin, 
        tmax=tmax,
        baseline=None,
        reject=None,
        picks=picks,
        preload=True,
    )
    ## Epoching for save
    epochs = mne.Epochs(
        preproc,
        events=events,
        event_id=event_id,
        tmin=tmin, 
        tmax=tmax,
        baseline=None,
        reject=None,
        picks=picks,
        preload=True,
    )
    ## Plot evoked for each condition
    evokeds = []
    for cond in ['Freq', 'Rare', 'Resp']:
        evoked = epochs[cond].average()
        fig = evoked.plot(show=False)
        report.add_figure(fig, title="Evoked ({})".format(cond))
        fig = evoked.plot_joint(show=False)
        report.add_figure(fig, title="Evoked ({}) - Joint".format(cond))
        evokeds.append(evoked)
    report.add_evokeds(evokeds, titles=["Evoked (Freq)", "Evoked (Rare)", "Evoked (Resp)"])

    ## First, run AR on the filtered data
    ar = AutoReject(picks='mag', n_jobs=-1)
    ar.fit(epochs_filt)
    autoreject_log = ar.get_reject_log(epochs_filt)
    print(np.sum(autoreject_log.bad_epochs))
    
    try:
        fig = epochs[autoreject_log.bad_epochs].plot()
        report.add_figure(fig, title="Bad pochs")
    except:
        print("No bad epochs")
    fig = autoreject_log.plot('horizontal')
    report.add_figure(fig, title="Autoreject decisions")

    ## Then, run ICA on the filtered data
    record_date = raw.info['meas_date'].strftime('%Y%m%d')
    noise_cov = load_noise_cov(record_date)
    report.add_covariance(noise_cov, info=preproc.info, title="Noise covariance matrix")

    ## Fit ICA without bad epochs
    ica = ICA(n_components=20, 
                random_state=0,#).fit(raw_filt, decim=3)
                noise_cov=noise_cov).fit(epochs_filt[~autoreject_log.bad_epochs], decim=3)
    fig = ica.plot_sources(preproc, show=True)
    report.add_figure(fig, title="ICA sources")

    ## Find ECG components
    ecg_threshold = 0.50
    ecg_epochs = create_ecg_epochs(preproc, ch_name="ECG")
    ecg_inds, ecg_scores = ica.find_bads_ecg(
        ecg_epochs, ch_name="ECG", method="ctps", threshold=ecg_threshold
    )
    if ecg_inds == []:
        ecg_inds = [list(abs(ecg_scores)).index(max(abs(ecg_scores)))]
    fig = ica.plot_scores(ecg_scores, ecg_inds, show=False)
    report.add_figure(fig, title="ECG scores")
    try:
        fig = ica.plot_properties(
            ecg_epochs, picks=ecg_inds, image_args={"sigma": 1.0}, show=False
        )
        for i, figure in enumerate(fig):
            report.add_figure(figure, title="Detected component " + str(i))
    except:
        print("No component to remove")

    ## Find EOG components
    eog_threshold = 4
    eog_epochs = create_eog_epochs(preproc, ch_name="vEOG")
    eog_inds, eog_scores = ica.find_bads_eog(
        eog_epochs, ch_name="vEOG", threshold=eog_threshold
    )
    if eog_inds == []:
        eog_inds = [list(abs(eog_scores)).index(max(abs(eog_scores)))]
    fig = ica.plot_scores(eog_scores, eog_inds, show=False)
    report.add_figure(fig, title="EOG scores")
    fig = list()
    try:
        fig = ica.plot_properties(
            eog_epochs, picks=eog_inds, image_args={"sigma": 1.0}, show=False
        )
        for i, figure in enumerate(fig):
            report.add_figure(figure, title="Detected component " + str(i))
            #close(figure)
    except:
        print("No component to remove")

    ## Exclude components
    to_remove = ecg_inds + eog_inds
    ica.exclude = to_remove
    preproc = ica.apply(preproc)
    epochs = ica.apply(epochs)

    ## Transform data with autoreject thresholds
    epochs = ar.fit_transform(epochs)
    autoreject_log = ar.get_reject_log(epochs)

    ## Plot cleaned signal
    fig = preproc.plot(duration=20, start=50, show=False)
    report.add_figure(fig, title="Time series (cleaned)")
    fig = preproc.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD (cleaned)")
    fig = epochs.plot(show=False)
    report.add_figure(fig, title="Epochs (cleaned)")
    fig = epochs.plot_psd(average=False, picks="mag", show=False)
    report.add_figure(fig, title="PSD (cleaned)")
    
    ## Plot evoked for each cond
    evokeds = []
    for cond in ['Freq', 'Rare', 'Resp']:
        evoked = epochs[cond].average()
        fig = evoked.plot(show=False)
        report.add_figure(fig, title="Evoked ({})".format(cond))
        fig = evoked.plot_joint(show=False)
        report.add_figure(fig, title="Evoked ({}) - Joint".format(cond))
        evokeds.append(evoked)
    ## Plot difference waves Rare - Freq
    evoked_diff = mne.combine_evoked([epochs['Rare'].average(), 
                                    -epochs['Freq'].average()], 
                                    weights='equal')
    fig = evoked_diff.plot(show=False)
    report.add_figure(fig, title="Evoked (Rare - Freq)")
    fig = evoked_diff.plot_joint(show=False)
    report.add_figure(fig, title="Evoked (Rare - Freq) - Joint")
    evokeds.append(evoked_diff)
    report.add_evokeds(evokeds, titles=["Evoked (Freq)", "Evoked (Rare)", "Evoked (Resp)", "Evoked (Rare - Freq)"])

    ## Set annotations for preproc
    preproc.set_annotations(events=mne.annotations_from_events(events), 
                            sfreq=raw.info['sfreq'],
                            event_desc={v: k for k, v in event_id.items()})
    del ar
    del ica
    del raw
    del raw_filt
    return epochs, preproc, autoreject_log, report

if __name__ == "__main__":
    args = parser.parse_args()
    subj = args.subject

    tmin=0.426 
    tmax=1.278

    for bloc in BLOCS_LIST:
        bloc = '0'+bloc
        # Create filenames
        filepaths = create_fnames(subj, bloc)
        if not os.path.isfile(str(filepaths['preproc'].fpath)):
            epochs, preproc, autoreject_log, report = preproc_pipeline(filepaths, tmin, tmax)
            ## Save preproc
            write_raw_bids(preproc, filepaths['preproc'], format='FIF', overwrite=True, allow_preload=True)
            ## Save epochs
            write_raw_bids(preproc, filepaths['epoch'], format='FIF', overwrite=True, allow_preload=True) # Init bids structure
            epochs.save(filepaths['epoch'].fpath, overwrite=True)
            
            ## Save AR log
            with open(str(filepaths['ARlog'].fpath)+'.pkl', 'wb') as f:
                pickle.dump(autoreject_log, f)

            ## Save report
            report.save(str(filepaths['report'].fpath)+'.html', open_browser=False, overwrite=True)
            ## Del objects to free some memory
            del epochs
            del preproc
            del autoreject_log
            del report