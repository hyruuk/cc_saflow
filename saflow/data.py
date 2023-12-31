import saflow
from saflow.stats import subject_average, simple_contrast, subject_contrast
import numpy as np
import pandas as pd
import os.path as op
import pickle
import pickle as pkl
import random
import os
from saflow.neuro import average_bands
from scipy.stats import zscore

def load_fooof_data(feature, feature_fpath, feat_to_get, type_how='alltrials', classif='INOUT_2575'):
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
                # Get events dicts
                events_dicts = get_events_dicts_from_magic_dict(data_reshaped)
                # Get VTC bounds
                if 'INOUT' in classif:
                    lowbound = int(classif.split('_')[1][:2])
                    highbound = int(classif.split('_')[1][2:])
                else:
                    lowbound = 50
                    highbound = 50
                inbound, outbound = get_VTC_bounds(events_dicts, lowbound=lowbound, highbound=highbound)

                for trial_idx in range(data_reshaped.shape[0]):
                    event_dict = events_dicts[trial_idx]
                    inout_epoch = get_inout(event_dict, inbound, outbound)
                    epoch_selected = select_epoch(event_dict, bad_how='any', type_how=type_how, inout_epoch=inout_epoch, verbose=False)
                    if epoch_selected:
                        trial_data = get_trial_data(data_reshaped, trial_idx, feat_to_get)
                        if not np.isnan(trial_data).any():
                            X.append(trial_data)
                            groups.append(sub)
                            VTC.append(np.nanmean(data_reshaped[trial_idx][0]['info']['included_VTC']))
                            task.append(data_reshaped[trial_idx][0]['info']['task'])
                            if 'INOUT' in classif:
                                # 0 = IN, 1 = OUT
                                if inout_epoch == 'IN':
                                    y.append(0)
                                elif inout_epoch == 'OUT':
                                    y.append(1)
                            elif classif == 'oddball':
                                # 0 = rare, 1 = frequent
                                if event_dict['task'] == 'correct_omission':
                                    y.append(0)
                                elif event_dict['task'] == 'correct_commission':
                                    y.append(1)
                            elif classif == 'rare':
                                # 0 = correct, 1 = error
                                if event_dict['task'] == 'correct_omission':
                                    y.append(0)
                                elif event_dict['task'] == 'commission_error':
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

def get_events_dicts_from_magic_dict(data_reshaped):
    events_dicts = []
    for trial_idx in range(data_reshaped.shape[0]):
        events_dicts.append(data_reshaped[trial_idx][0]['info'])
    return events_dicts


def load_features(feature_folder, subject='all', feature='psd', splitby='inout', inout='INOUT_2575', remove_errors=True, get_task = ['correct_commission']):
    ''' 0 = IN, 1 = OUT
    '''
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
                epoch_selected = select_trial(data['info'], type_how='lapse', inout=inout, verbose=True)
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
    ## TODO : dépister potentiel bug avec IN et OUT
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



def balance_dataset(X, y, groups, seed=42069):
    # Number of observations
    n_observations = X.shape[1]

    # Initialize result lists
    balanced_indices = []

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Process each group
    for group in np.unique(groups):
        # Indices for the current group
        group_indices = np.where(groups == group)[0]

        # Count observations in each class within the group
        class_counts = {label: np.sum(y[group_indices] == label) for label in np.unique(y[group_indices])}
        # Print the class counts for the current group
        print('Group {}: {}'.format(group, class_counts))

        # Find the minimum count among classes in this group
        min_count = min(class_counts.values())

        # Sample from each class
        for class_value in np.unique(y[group_indices]):
            class_indices = group_indices[y[group_indices] == class_value]
            sampled_indices = np.random.choice(class_indices, size=min_count, replace=False)
            balanced_indices.extend(sampled_indices)

    # Sort indices to maintain original order
    balanced_indices = sorted(set(balanced_indices))

    # Select balanced data
    balanced_X = X[:, balanced_indices, :]
    balanced_y = y[balanced_indices]
    balanced_groups = groups[balanced_indices]
    return balanced_X, balanced_y, balanced_groups


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
    
    elif type_how == 'alltrials':
        retain_type = True

    retain_epoch = retain_inout & retain_type & ~bad_epoch
    if verbose:
        print(event_dict['included_task'])
        print(f'Bad : {bad_epoch}, InOut : {retain_inout}, Type : {retain_type}, Retain : {retain_epoch}')
    return retain_epoch


def get_VTC_bounds(events_dicts, lowbound=25, highbound=75):
    # get the averaged VTC for each epoch
    run_VTCs = []
    for info_dict in events_dicts:
        run_VTCs.append(np.mean(info_dict['included_VTC'], axis=0))
    
    # obtain the bounds of the VTC for this run
    inbound = np.percentile(run_VTCs, lowbound, axis=0)
    outbound = np.percentile(run_VTCs, highbound, axis=0)
    return inbound, outbound

def get_inout(info_dict, inbound, outbound):
    VTC = np.mean(info_dict['included_VTC'], axis=0)
    if VTC <= inbound:
        return 'IN'
    elif VTC >= outbound:
        return 'OUT'
    else:
        return 'MID'

def select_epoch(event_dict, bad_how='any', type_how='alltrials', inout_epoch=None, verbose=False):
    # Check if bad epoch
    if bad_how == 'any':
        bad_epoch = np.sum(event_dict['included_bad_epochs']) > 0
    else:
        bad_epoch = False

    # Check trial types across the epoch
    if type_how == 'alltrials':
        retain_task = True
    elif type_how == 'correct':
        correct_task_types = ['correct_omission', 'correct_commission']
        retain_task = all(item in correct_task_types for item in event_dict['included_task'])
    elif type_how == 'lapse':
        retain_task = 'commission_error' in event_dict['included_task']
    
    # Check inout type
    retain_inout = False
    if inout_epoch is not None:    
        if inout_epoch in ['IN', 'OUT']:
            retain_inout = True
    
    retain_epoch = retain_task & ~bad_epoch & retain_inout

    if verbose:
        print(f'Bad : {bad_epoch}, InOut : {retain_inout}, Type : {retain_task}, Retain : {retain_epoch}')
        print(event_dict['included_task'])
        print(event_dict['included_bad_epochs'])
    return retain_epoch