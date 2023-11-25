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
        runs = ['0' + x for x in saflow.BLOCS_LIST]
    else:
        runs = [run]

    
    fg = FOOOFGroup(aperiodic_mode=method, max_n_peaks=8)
    for subject in subjects:
        IN_subj = []
        OUT_subj = []
        for run in runs:
            print(f'Processing subject {subject}, run {run}')
            filepaths = create_fnames(subject, run)
            input_fname = str(filepaths['welch'].fpath) + '.pkl'
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
            0/0
            # Run-level FOOOFs
            fg_IN = fg.copy()
            fg_IN.fit(freq_bins, IN_run_avg, [2,120], n_jobs=n_jobs)
            fg_OUT = fg.copy()
            fg_OUT.fit(freq_bins, OUT_run_avg, [2,120], n_jobs=n_jobs)
            
            # Trial-level FOOOFs
            trial_fooofs = []
            for idx_trial, trial in enumerate(welch_array):
                print(f'Processing trial {idx_trial} for subject-{subject} run-{run}')
                fg_trial = fg.copy()
                fg_trial.fit(freq_bins, trial, [2,120], n_jobs=n_jobs)
                trial_fooofs.append(fg_trial)
            #trial_fooofs = np.array(trial_fooofs)

            # Save
            filepaths['welch'].update(root=saflow.BIDS_PATH + '/derivatives/fooof/').mkdir(exist_ok=True)
            output_fname = str(filepaths['welch'].update(root=saflow.BIDS_PATH + '/derivatives/fooof/').fpath) + '.pkl'
            with open(output_fname, 'wb') as f:
                pickle.dump({'IN_fooofs': fg_IN,
                            'OUT_fooofs': fg_OUT,
                            'trial_fooofs': trial_fooofs,
                            'info': events_dicts}, f)
                
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
        output_fname = str(filepaths['welch'].update(root=saflow.BIDS_PATH + '/derivatives/fooof/', run=None).fpath) + '.pkl'
        print(output_fname)

        with open(output_fname, 'wb') as f:
            pickle.dump({'IN_fooofs': fg_subj_IN,
                        'OUT_fooofs': fg_subj_OUT}, f)
