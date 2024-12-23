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
from saflow.data import select_epoch, get_VTC_bounds, get_inout
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
    default="fixed",
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
    default='2044_atlas_aparc_sub_8trials_selfcorr',
    type=str,
    help="Welch from which to process",
)

parser.add_argument(
    "-t",
    "--type",
    default='alltrials',
    type=str,
    help="Type of trials to consider",
)
parser.add_argument(
    "-b",
    "--bounds",
    default='2575',
    type=str,
    help="Bounds of the VTC window",
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
        subjects = [x.split('-')[1] for x in os.listdir(op.join(saflow.BIDS_PATH, 'derivatives', 'welch_' + welch_params))]
    else:
        subjects = [subjects]
    if run == 'all':
        runs = ['0' + x for x in saflow.BLOCS_LIST]
    else:
        runs = [run]

    type_how = args.type
    lowbound = int(args.bounds[:2])
    highbound = int(args.bounds[2:])
    
    fooof_params = f'fooof_fixed_{welch_params}'

    max_n_peaks = 8
    peak_width_limits = [2, 12]
    fg = FOOOFGroup(aperiodic_mode=method, peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks)
    for subject in subjects:
        IN_subj = []
        OUT_subj = []
        runs = [x.split('_')[2].split('-')[1] for x in os.listdir(op.join(saflow.BIDS_PATH, 'derivatives', 'welch_' + welch_params, 'sub-' + subject, 'meg'))]
        for run in runs:
            print(f'Processing subject {subject}, run {run}')
            filepaths = create_fnames(subject, run)
            input_fname = filepaths['welch'].update(root=op.join('/'.join(str(filepaths['welch'].root).split('/')[:-1]), str(filepaths['welch'].root).split('/')[-1] + f'_{welch_params}'))
            input_fname = str(filepaths['welch'].fpath) + '.pkl'
            if 'aparc' in welch_params:
                input_fname = input_fname.replace('.pkl', '_aparc_sub.pkl')
            print(input_fname)
            IN_baseline = []
            OUT_baseline = []

            # Load data
            with open(input_fname, 'rb') as f:
                file = pickle.load(f)
                welch_array = file['data']
                events_dicts = file['info']
                freq_bins = file['freq_bins']

            inbound, outbound = get_VTC_bounds(events_dicts, lowbound=25, highbound=75)
            # Grab correct baseline trials
            for idx_trial, trial in enumerate(welch_array):
                event_dict = events_dicts[idx_trial]
                inout_epoch = get_inout(event_dict, inbound, outbound)
                epoch_selected = select_epoch(event_dict, bad_how='any', type_how=type_how, inout_epoch=inout_epoch, verbose=True)
                if epoch_selected:
                    if inout_epoch == 'IN':
                        IN_baseline.append(trial)
                    elif inout_epoch == 'OUT':
                        OUT_baseline.append(trial)

            if len(IN_baseline) != 0 and len(OUT_baseline) != 0:
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
        filepaths['welch'].mkdir(exist_ok=True)
        print(output_fname)

        with open(output_fname, 'wb') as f:
            pickle.dump({'IN_fooofs': fg_subj_IN,
                        'OUT_fooofs': fg_subj_OUT}, f)