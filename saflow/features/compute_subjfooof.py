import saflow
from saflow.utils import create_fnames, segment_sourcelevel
import os.path as op
import os
import argparse
import pickle
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from fooof import FOOOF, Bands, FOOOFGroup
import mne_bids
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    default="knee",
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
parser.add_argument(
    "-wp",
    "--welch_params",
    default='1022_sensor_8trials',
    type=str,
    help="Welch from which to process",
)




if __name__ == "__main__":
    args = parser.parse_args()
    n_jobs = args.n_jobs
    n_trials = args.n_trials
    level = args.level
    method = args.method
    subjects = args.subject
    run = args.run
    chan = args.channel
    welch_params = args.welch_params
    if subjects == 'all':
        subjects = saflow.SUBJ_LIST
    else:
        subjects = [subjects]
    if run == 'all':
        runs = ['0' + x for x in saflow.BLOCS_LIST]
    else:
        runs = [run]

    
    fooof_params = 'fooof_1022_knee_sensor_8trials'

    max_n_peaks = 8
    peak_width_limits = [2, 12]
    fg = FOOOFGroup(aperiodic_mode=method, peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks)
    for subject in subjects:
        IN_subj = []
        OUT_subj = []
        for run in runs:
            print(f'Processing subject {subject}, run {run}')
            filepaths = create_fnames(subject, run)
            input_fname = filepaths['welch'].update(root=op.join('/'.join(str(filepaths['welch'].root).split('/')[:-1]), str(filepaths['welch'].root).split('/')[-1] + f'_{welch_params}'))
            input_fname = str(filepaths['welch'].fpath) + '.pkl'
            print(input_fname)
            IN_baseline = []
            OUT_baseline = []

            # Load data
            with open(input_fname, 'rb') as f:
                file = pickle.load(f)
                welch_array = file['data']
                events_dicts = file['info']
                freq_bins = file['freq_bins']

            # Grab correct baseline trials
            for idx_trial, trial in enumerate(welch_array):
                if events_dicts[idx_trial]['bad_epoch'] == False:
                    if events_dicts[idx_trial]['task'] == 'correct_commission':
                        if events_dicts[idx_trial]['INOUT_2575'] == 'IN':
                            IN_baseline.append(trial)
                        elif events_dicts[idx_trial]['INOUT_2575'] == 'OUT':
                            OUT_baseline.append(trial)

            # Compute trial-averages
            IN_run_avg = np.mean(np.array(IN_baseline), axis=0)
            OUT_run_avg = np.mean(np.array(OUT_baseline), axis=0)
            IN_subj.append(IN_run_avg)
            OUT_subj.append(OUT_run_avg)

        # Subject-level FOOOFs        
        IN_subj = np.array(IN_subj)
        OUT_subj = np.array(OUT_subj)

        # Average across runs
        IN_subj_avg = np.mean(IN_subj, axis=0)
        OUT_subj_avg = np.mean(OUT_subj, axis=0)

        # Compute FOOOFs
        fg_subj_IN = fg.copy()
        fg_subj_IN.fit(freq_bins, IN_subj_avg, [2,120], n_jobs=n_jobs)
        fg_subj_OUT = fg.copy()
        fg_subj_OUT.fit(freq_bins, OUT_subj_avg, [2,120], n_jobs=n_jobs)

        # Save
        output_fname = str(filepaths['welch'].update(root=saflow.BIDS_PATH + f'/derivatives/{fooof_params}/', run=None).fpath) + '.pkl'
        print(output_fname)

        with open(output_fname, 'wb') as f:
            pickle.dump({'IN_fooofs': fg_subj_IN,
                        'OUT_fooofs': fg_subj_OUT}, f)