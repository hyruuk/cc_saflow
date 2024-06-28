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
    default="atlas_aparc_sub",
    type=str,
    help="Level of processing (sensor or source or atlas_atlas-name)",
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


if __name__ == "__main__":
    args = parser.parse_args()
    n_jobs = args.n_jobs
    n_trials = args.n_trials
    level = args.level
    method = args.method
    subj = args.subject
    run = args.run

    n_fft = 2044
    n_overlap = 1533

    if subj == 'all':
        subjects = saflow.SUBJ_LIST
    else:
        subjects = [subj]
    if run == 'all':
        runs = ['0' + x for x in saflow.BLOCS_LIST]
    else:
        runs = [run]

    
    
    for subject in subjects:
        for run in runs:
            try:
                print(f'Processing subject {subject}, run {run}')
                from saflow.utils import create_fnames
                filepaths = create_fnames(subject, run)
                filepaths['welch'].update(root=op.join('/'.join(str(filepaths['welch'].root).split('/')[:-1]), str(filepaths['welch'].root).split('/')[-1] + f'_{n_fft}_{level}_{n_trials}trials'))
                filepaths['welch'].mkdir(exist_ok=True)
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
                    
                    elif 'atlas' in level:
                        if 'aparc' in level:
                            atlas_name = 'aparc' + level.split('aparc')[1]
                        else:
                            atlas_name = level.split('_')[1]

                        from saflow.source_reconstruction.apply_atlas import create_fnames
                        atlas_filepaths = create_fnames(subject, run, 
                                                f'morphed_sources_{atlas_name}', 
                                                f'welch_{atlas_name}')
                        with open(str(atlas_filepaths['input'].fpath).replace('desc-morphed', 'desc-atlased-avg') + '.pkl', 'rb') as f:
                            file_content = pickle.load(f)
                            data = file_content['data']
                            sfreq = file_content['sfreq']
                            region_names = file_content['region_names']
                        output_fname = output_fname.replace('.pkl', f'_{atlas_name}.pkl')
                        print(data.shape)
                        
                    segmented_array, events_idx, events_dicts = segment_sourcelevel(data, filepaths, sfreq=sfreq, n_events_window=n_trials)
                    welch_array, freq_bins = mne.time_frequency.psd_array_welch(segmented_array, sfreq=sfreq, n_jobs=n_jobs, n_fft=n_fft, n_overlap=n_overlap, average='mean')
                    
                    with open(output_fname, 'wb') as f:
                        pickle.dump({'data':welch_array,
                                    'info':events_dicts,
                                    'freq_bins':freq_bins}, f)
            except Exception as e:
                print(e)
                continue