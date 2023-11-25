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

def compute_fooofs_on_averages(avg_psd, n_jobs=-1, method='knee', max_n_peaks=8, output_fname=None):
    fg = FOOOFGroup(aperiodic_mode=method, max_n_peaks=max_n_peaks)
    count = 0
    fooof_groups = []
    conditions_list = []
    subjects_list = []
    for cond_idx, cond in enumerate(['IN', 'OUT']):
        for subj_idx in range(len(avg_psd['IN'])):
            count = count + 1
            print('=====================================')
            print(f'Processing group {count} for cond {cond}')
            fg_temp = fg.copy()
            fg_temp.fit(avg_psd['freq_bins'], avg_psd[cond][subj_idx], [2,120], n_jobs=n_jobs)
            fooof_groups.append(fg_temp)
            conditions_list.append(cond)
            subjects_list.append(avg_psd['subject_list'][subj_idx])
    fooof_array = np.array(fooof_groups)
    data_dict = {'fooof_array': fooof_array,
                 'conditions': conditions_list,
                'subjects': subjects_list}
    if output_fname is not None:
        with open(output_fname, 'wb') as f:
            pickle.dump(data_dict, f)
    return fooof_array, conditions_list, subjects_list


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
            print(f'Processing subject {subject}, run {run}')
            run = '0' + str(run)
            filepaths = create_fnames(subject, run)
            output_fname = str(filepaths['welch'].fpath) + '.pkl'
            if not op.exists(output_fname):
                subject_list = []
                run_list = []
                trial_psd = []
                info_list = []

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
                
                with open(output_fname, 'wb') as f:
                    pickle.dump({'data':welch_array,
                                'info':events_dicts,
                                'freq_bins':freq_bins}, f)


'''
    ## Compute FOOOFs
    output_fname = op.join(saflow.BIDS_PATH, 'derivatives', 'subject_averaged_fooofs_knee-mp8.pkl')
    if not op.exists(output_fname):
        fg = FOOOFGroup(aperiodic_mode='knee')
        count = 0
        fooof_groups = []
        conditions_list = []
        subjects_list = []
        for cond_idx, cond in enumerate(['IN', 'OUT']):
            for subj_idx in range(len(avg_psd['IN'])):
                count = count + 1
                print('=====================================')
                print(f'Processing group {count} for cond {cond}')
                fg_temp = fg.copy()
                fg_temp.fit(avg_psd['freq_bins'], avg_psd[cond][subj_idx], [2,120], n_jobs=n_jobs)
                fooof_groups.append(fg_temp)
                conditions_list.append(cond)
                subjects_list.append(avg_psd['subject_list'][subj_idx])
        fooof_array = np.array(fooof_groups)
        data_dict = {'fooof_array': fooof_array,
                     'conditions': conditions_list,
                    'subjects': subjects_list}
                
        with open(output_fname, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        with open(output_fname, 'rb') as f:
            data_dict = pickle.load(f)
            fooof_array = data_dict['fooof_array']
            conditions_list = data_dict['conditions']
            subjects_list = data_dict['subjects']
'''


