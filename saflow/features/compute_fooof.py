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
    default='1022_sensor_8trials',
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
        subjects = saflow.SUBJ_LIST
    else:
        subjects = [subjects]
    if run == 'all':
        runs = ['0' + x for x in saflow.BLOCS_LIST]
    else:
        runs = [run]

    type_how = args.type
    lowbound = int(args.bounds[:2])
    highbound = int(args.bounds[2:])

    fooof_params = f'fooof_{method}_{type_how}_{welch_params}'

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

            inbound, outbound = get_VTC_bounds(events_dicts, lowbound=lowbound, highbound=highbound)
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
            print(f'Found {len(IN_baseline)} IN trials and {len(OUT_baseline)} OUT trials for subject {subject} run {run}')
            # Compute trial-averages
            IN_run_avg = np.mean(np.array(IN_baseline), axis=0)
            OUT_run_avg = np.mean(np.array(OUT_baseline), axis=0)
            IN_subj.append(IN_run_avg)
            OUT_subj.append(OUT_run_avg)

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

            # Save
            filepaths['welch'].update(root=saflow.BIDS_PATH + f'/derivatives/{fooof_params}/').mkdir(exist_ok=True)
            output_fname = str(filepaths['welch'].update(root=saflow.BIDS_PATH + f'/derivatives/{fooof_params}_selfcorr/').fpath) + '.pkl'
            output_dict = {'IN_fooofs': fg_IN,
                            'OUT_fooofs': fg_OUT,
                            'trial_fooofs': trial_fooofs,
                            'info': events_dicts}
            with open(output_fname, 'wb') as f:
                pickle.dump(output_dict, f)
            
            data = output_dict
            # Create magic dict
            magic_dict = []
            for trial_idx in range(len(data['trial_fooofs'])):
                INOUT = data['info'][trial_idx]['INOUT']
                print(f'Adding trial {trial_idx} of run {run} of subject {subject}')
                for chan_idx in range(len(data['trial_fooofs'][trial_idx])):
                    psd_raw = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).power_spectrum
                    freq_bins = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).freqs
                    
                    # currently deprec
                    #if INOUT == 'IN':
                    #    psd_corrected = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).power_spectrum - data['IN_fooofs'].get_fooof(chan_idx)._ap_fit
                    #elif INOUT == 'OUT':
                    #    psd_corrected = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).power_spectrum - data['OUT_fooofs'].get_fooof(chan_idx)._ap_fit
                    # now live
                    psd_corrected = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).power_spectrum - data['trial_fooofs'][trial_idx].get_fooof(chan_idx)._ap_fit

                    psd_model_fit = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).fooofed_spectrum_
                    exponent = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).aperiodic_params_[-1]
                    offset = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).aperiodic_params_[0]
                    knee = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).aperiodic_params_[1]
                    r_squared = data['trial_fooofs'][trial_idx].get_fooof(chan_idx).r_squared_
                    
                    data_dict = {'psd_raw': psd_raw, 
                                'psd_corrected': psd_corrected, 
                                'psd_model_fit': psd_model_fit, 
                                'exponent': exponent, 
                                'offset': offset, 
                                'knee': knee, 
                                'r_squared': r_squared,
                                'info': data['info'][trial_idx],
                                'freq_bins': freq_bins}
                    magic_dict.append(data_dict)
            fname_output = output_fname.replace('.pkl', '_magic.pkl')
            with open(fname_output, 'wb') as f:
                pickle.dump(magic_dict, f)
