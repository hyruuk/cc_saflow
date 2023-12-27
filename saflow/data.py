import saflow
from saflow.stats import subject_average, simple_contrast, subject_contrast
import numpy as np
import os.path as op
import pickle
import pickle as pkl
import random
import os
from saflow.neuro import average_bands
from scipy.stats import zscore

def load_fooof_data(feature, feature_fpath, feat_to_get, trial_type_to_get, classif='INOUT_2575'):
    if type(trial_type_to_get) == str:
        trial_type_to_get = [trial_type_to_get]
    
    X = []
    y = []
    groups = []
    VTC = []
    task = []
    for sub in sorted(os.listdir(feature_fpath)):
        for file in sorted(os.listdir(op.join(feature_fpath, sub, 'meg'))):
            print(file)
            if 'magic' in file:
                fname = op.join(feature_fpath, sub, 'meg', file)
                fname_output = fname.replace('.pkl', '_magic.pkl')
                with open(fname, 'rb') as f:
                    data = pickle.load(f)
                # Reshape data
                n_chans = 270
                n_trials = int(len(data)/n_chans)
                data_reshaped = np.array(data).reshape(n_trials, n_chans)

                for trial_idx in range(data_reshaped.shape[0]):
                    epoch_selected = select_trial(data_reshaped[trial_idx][0]['info'], inout=classif, type_how='lapse')
                    if epoch_selected:
                        trial_data = get_trial_data(data_reshaped, trial_idx, feat_to_get)
                        if not np.isnan(trial_data).any():
                            X.append(trial_data)
                            groups.append(sub)
                            VTC.append(np.nanmean(data_reshaped[trial_idx][0]['info']['included_VTC']))
                            task.append(data_reshaped[trial_idx][0]['info']['task'])
                            if 'INOUT' in classif:
                                # 0 = IN, 1 = OUT
                                if data_reshaped[trial_idx][0]['info'][classif] == 'IN':
                                    y.append(0)
                                elif data_reshaped[trial_idx][0]['info'][classif] == 'OUT':
                                    y.append(1)
                            elif classif == 'oddball':
                                # 0 = rare, 1 = frequent
                                if data_reshaped[trial_idx][0]['info']['task'] == 'correct_omission':
                                    y.append(0)
                                elif data_reshaped[trial_idx][0]['info']['task'] == 'correct_commission':
                                    y.append(1)
                        else:
                            print('Nan in trial data')
                            print(f'{sub} {file}, trial {trial_idx}')
    X = np.array(X).transpose(2,0,1)
    y = np.array(y)
    groups = np.array(groups)
    VTC = np.array(VTC)
    task = np.array(task)
    return X, y, groups, VTC, task

def load_features(feature_folder, subject='all', feature='psd', splitby='inout', inout='INOUT_2575', remove_errors=True, get_task = ['correct_commission']):
    ''' 0 = IN, 1 = OUT
    '''
    subject
    X = []
    y = []
    groups = []
    VTC = []
    task = []
    for idx_subj, subj in enumerate(sorted(os.listdir(feature_folder))):
        for trial in os.listdir(op.join(feature_folder, subj, 'meg')):
            if 'desc-' in trial:
                filepath = op.join(feature_folder, subj, 'meg', trial)
                with open(filepath, 'rb') as f:
                    data = pkl.load(f)
                print(trial)
                epoch_selected = select_trial(data['info'], type_how='correct', inout=inout)
                if epoch_selected:
                    try:
                        if splitby == 'inout':
                            if type(data['info'][inout]) == str:
                                if data['info'][inout] == 'IN':
                                    y.append(0)
                                elif data['info'][inout] == 'OUT':
                                    y.append(1)
                                groups.append(idx_subj)

                                if feature in ['psd', 'lzc']:
                                    X.append(data['data'])

                                elif feature in ['slope', 'offset', 'r_squared', 'knee']:
                                    with open(filepath.replace('_desc-', '_fg-'), 'rb') as f:
                                        fooof_data = pkl.load(f)
                                    temp_X = []
                                    for fm in fooof_data['fooof']:
                                        if feature == 'slope':
                                            temp_X.append(fm.get_params('aperiodic_params')[-1])
                                        elif feature == 'offset':
                                            temp_X.append(fm.get_params('aperiodic_params')[0])
                                        elif feature == 'knee':
                                            temp_X.append(fm.get_params('aperiodic_params')[1])
                                        elif feature == 'r_squared':
                                            temp_X.append(fm.get_params('r_squared'))
                                    X.append(np.array(temp_X))

                                VTC.append(np.nanmean(data['info']['included_VTC']))
                                task.append(data['info']['task'])
                    except Exception as e:
                        print(f'Could not load {trial}')
                        print(e)
    X = np.array(X)
    if X.ndim == 3:
        X = X.transpose(1,0,2)
    y = np.array(y)
    groups = np.array(groups)
    VTC = np.array(VTC)
    task = np.array(task)
    return X, y, groups, {'vtc':VTC, 'task':task}

def load_subjlevel_fooofs(feature, feature_fpath, zscored=False):
    X_raw = []
    X_corrected = []
    X_model = []
    X_ksor = []
    y = []
    groups = []
    n_chans = 270
    for sub_idx, sub in enumerate(sorted(os.listdir(feature_fpath))):
        for file in sorted(os.listdir(op.join(feature_fpath, sub, 'meg'))):
            if 'run' not in file:
                fpath = op.join(feature_fpath, sub, 'meg', file)
                with open(fpath, 'rb') as f:
                    fooof_dict = pkl.load(f)
                for condition in ['IN_fooofs', 'OUT_fooofs']:
                    psds_raw = []
                    psds_corrected = []
                    psds_model = []
                    ksor_list = []
                    for chan_idx in range(n_chans):
                        fm = fooof_dict[condition].get_fooof(chan_idx)
                        psd_raw = fm.power_spectrum
                        psd_corrected = fm.power_spectrum - fm._ap_fit
                        psd_model = fm.fooofed_spectrum_
                        ksor = [fm.get_params('aperiodic_params')[-1], 
                                fm.get_params('aperiodic_params')[0], 
                                fm.get_params('aperiodic_params')[1], 
                                fm.get_params('r_squared')]
                        freq_bins = fm.freqs
                        psds_raw.append(average_bands(psd_raw, freq_bins))
                        psds_corrected.append(average_bands(psd_corrected, freq_bins))
                        psds_model.append(average_bands(psd_model, freq_bins))
                        ksor_list.append(np.array(ksor))
                    if zscored:
                        X_raw.append(zscore(np.array(psds_raw), axis=0))
                        X_corrected.append(zscore(np.array(psds_corrected), axis=0))
                        X_model.append(zscore(np.array(psds_model), axis=0))
                        X_ksor.append(zscore(np.array(ksor_list), axis=0))
                    else:
                        X_raw.append(np.array(psds_raw))
                        X_corrected.append(np.array(psds_corrected))
                        X_model.append(np.array(psds_model))
                        X_ksor.append(np.array(ksor_list))
                    groups.append(sub_idx)
                    y.append(0 if condition == 'IN_fooofs' else 1)

    X_raw = np.array(X_raw).transpose(2,0,1)
    X_corrected = np.array(X_corrected).transpose(2,0,1)
    X_model = np.array(X_model).transpose(2,0,1)
    X_ksor = np.array(X_ksor).transpose(2,0,1)
    y = np.array(y)
    groups = np.array(groups)
    return X_raw, X_corrected, X_model, X_ksor, y, groups

def get_trial_data(data_reshaped, trial_idx, feat_to_get, freq_bins=None, zscored=False):
    trial_data = []
    for chan_idx in range(data_reshaped.shape[1]):
        if feat_to_get == 'ksor':
            feat_data = []
            feat_data.append(data_reshaped[trial_idx][chan_idx]['knee'])
            feat_data.append(data_reshaped[trial_idx][chan_idx]['exponent'])
            feat_data.append(data_reshaped[trial_idx][chan_idx]['offset'])
            feat_data.append(data_reshaped[trial_idx][chan_idx]['r_squared'])
            trial_data.append(np.array(feat_data))
        else:
            psd = data_reshaped[trial_idx][chan_idx][feat_to_get]
            if freq_bins is None:
                freq_bins = data_reshaped[trial_idx][chan_idx]['freq_bins']
            power_bands = average_bands(psd, freq_bins)
            trial_data.append(power_bands)
    if zscored:
        trial_data = zscore(np.array(trial_data), axis=0)
    else:
        trial_data = np.array(trial_data)
    return trial_data


def balance_data(X, y, groups, seed=10):
    X_balanced = []
    y_balanced = []
    groups_balanced = []
    # We want to balance the trials across subjects
    random.seed(seed)
    X = X.transpose(1,0,2)
    for subj_idx in np.unique(groups):
        y_subj = [label for i, label in enumerate(y) if groups[i] == subj_idx]
        if len(np.unique(y_subj)) > 1:
            max_trials = min(np.unique(y_subj, return_counts=True)[1])
        
            X_subj_0 = [x for i, x in enumerate(X) if groups[i] == subj_idx and y[i] == 0]
            X_subj_1 = [x for i, x in enumerate(X) if groups[i] == subj_idx and y[i] == 1]

            idx_list_0 = [x for x in range(len(X_subj_0))]
            idx_list_1 = [x for x in range(len(X_subj_1))]
            inout = 'IN' if y_subj[0] == 0 else 'OUT'
            print(f'Subject {subj_idx} max trials : {max_trials} {inout}')
            print(len(idx_list_0), len(idx_list_1))
            print(max_trials)
            picks_0 = random.sample(idx_list_0, max_trials)
            picks_1 = random.sample(idx_list_1, max_trials)

            for i in range(max_trials):
                X_balanced.append(X_subj_0[picks_0[i]])
                y_balanced.append(0)
                groups_balanced.append(subj_idx)
                X_balanced.append(X_subj_1[picks_1[i]])
                y_balanced.append(1)
                groups_balanced.append(subj_idx)
        else:
            print(f'Not enough trials in one condition. Subject {subj_idx} removed.')
    X = np.array(X_balanced).transpose(1,0,2)
    y = np.array(y_balanced)
    groups = np.array(groups_balanced)
    return X, y, groups


def select_trial(event_dict, trial_type=['correct_commission'], type_how='correct', bad_how='any', inout_how='all', inout='INOUT_2575', verbose=False):
    if bad_how is None or bad_how == 'ignore':
        bad_epoch = False
    elif bad_how == 'last':
        bad_epoch = event_dict['bad_epoch']
    elif bad_how == 'any':
        bad_epoch = np.sum(event_dict['included_bad_epochs']) > 0

    if str(event_dict[inout]) != 'nan':
        if inout_how == 'all':
            n_uniques = len(np.unique(event_dict['included_INOUT'])) == 1
            unique_elements = np.unique(event_dict['included_INOUT'])
            uniques_inout = any(element in ['IN', 'OUT'] for element in unique_elements)
            retain_inout =  n_uniques & uniques_inout
        elif inout_how == 'last':
            retain_inout = event_dict[inout] in ['IN', 'OUT']
    else:
        retain_inout = False

    # deprec
    if type_how == 'last':
        retain_type = event_dict['task'] in trial_type
    if type_how == 'all':
        if len(np.unique(event_dict['included_task'])) == 1:
            if np.unique(event_dict['included_task']) == 'correct_omission':
                retain_type = True
            else:
                retain_type = False
        else:
            if event_dict['task'] in trial_type:
                retain_type = True
            else:
                retain_type = False

    # used
    if type_how == 'correct':
        correct_trials = ['correct_omission', 'correct_commission']
        retain_type = all(item in correct_trials for item in event_dict['included_task'])

    elif type_how == 'lapse':
        retain_type = 'commission_error' in event_dict['included_task']


    retain_epoch = retain_inout & retain_type & ~bad_epoch
    if verbose:
        print(f'Bad : {bad_epoch}, InOut : {retain_inout}, Type : {retain_type}, Retain : {retain_epoch}')
    return retain_epoch