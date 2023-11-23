import saflow
from saflow.features.utils import create_fnames, segment_sourcelevel
import os.path as op
import os
import argparse
import pickle
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
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
    help="Number of jobs to run in parallel",
)
parser.add_argument(
    "-c",
    "--channel",
    default='all',
    type=str,
    help="Channel to process",
)


if __name__ == "__main__":
    args = parser.parse_args()
    n_jobs = args.n_jobs
    n_trials = args.n_trials
    level = args.level
    method = args.method
    subj = args.subject
    run = args.run
    chan = args.channel
    if subj == 'all':
        subjects = saflow.SUBJ_LIST
    else:
        subjects = [subj]
    if run == 'all':
        runs = saflow.BLOCS_LIST
    else:
        runs = [run]


    for subject in subjects:
        for run in runs:
            filepaths = create_fnames(subject, run)
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

            segmented_array, events_idx, events_dicts = segment_sourcelevel(data, filepaths, sfreq=sfreq, n_events_window=n_trials)
            welch_array, freq_bins = mne.time_frequency.psd_array_welch(segmented_array, sfreq=sfreq, n_jobs=n_jobs, n_fft=1022, n_overlap=959, average='mean')

            0/0