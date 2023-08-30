# Computes inverse solution and saves sources
from saflow import FS_SUBJDIR, SUBJ_LIST, BLOCS_LIST, BIDS_PATH, invsol_params

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
import os
from mne_bids import BIDSPath
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='04',
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-r",
    "--run",
    default='02',
    type=str,
    help="Run to process",
)

def create_fnames(subject, bloc):
    # Setup input files
    preproc_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            suffix='meg',
                            processing='clean',
                            root=BIDS_PATH + '/derivatives/preprocessed/')

    epoch_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            processing='epo',
                            root=BIDS_PATH + '/derivatives/epochs/')
    
    # Find date for noise file
    info = mne.io.read_info(preproc_bidspath.fpath)
    er_date = info['meas_date'].strftime('%Y%m%d')
    # Noise file
    noise_bidspath = BIDSPath(subject='emptyroom', 
                        session=er_date,
                        task='noise', 
                        datatype="meg",
                        root=BIDS_PATH)
    
    # Setup output files
    bem_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='bem',
                            root=BIDS_PATH + '/derivatives/bem/')
    os.makedirs(os.path.dirname(bem_bidspath.fpath), exist_ok=True)
    
    # Coregistration transform
    trans_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            processing='trans',
                            root=BIDS_PATH + '/derivatives/trans/')
    os.makedirs(os.path.dirname(trans_bidspath.fpath), exist_ok=True)

    # Forward solution
    fwd_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='forward',
                            root=BIDS_PATH + '/derivatives/fwd/')
    os.makedirs(os.path.dirname(fwd_bidspath.fpath), exist_ok=True)

    # Noise covariance matrix
    noise_cov_bidspath = BIDSPath(subject='emptyroom', 
                    session=er_date,
                    task='noise', 
                    datatype="meg",
                    root=BIDS_PATH + '/derivatives/noise_cov/')
    os.makedirs(os.path.dirname(noise_cov_bidspath.fpath), exist_ok=True)

    # Sources
    stc_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='epo',
                            description='sources'
                            root=BIDS_PATH + '/derivatives/minimum-norm-estimate/')
    os.makedirs(os.path.dirname(stc_bidspath.fpath), exist_ok=True)

    # Morph
    morph_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='epo',
                            description='morphed',
                            root=BIDS_PATH + '/derivatives/minimum-norm-estimate/')

    return {'preproc':str(preproc_bidspath) + '.fif',
            'epoch':str(epoch_bidspath) + '.fif',
            'bem':str(bem_bidspath) + '.h5',
            'noise':str(noise_bidspath) + '.ds',
            'trans':str(trans_bidspath) + '.fif',
            'fwd':str(fwd_bidspath) + '.fif',
            'noise_cov':str(noise_cov_bidspath) + '.fif',
            'stc':str(stc_bidspath) + '.h5',
            'morph':str(morph_bidspath) + '.h5'}

def get_source_space(subject, subjects_dir=FS_SUBJDIR):
    # Compute Source Space
    return mne.setup_source_space(
                subject, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir
            )

def get_bem(subject):
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers (EEG)
    model = mne.make_bem_model(
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=FS_SUBJDIR
    )
    return mne.make_bem_solution(model)

def get_forward(filepath, src, bem):
    preproc_fullpath = filepath['preproc']
    trans_fullpath = filepath['trans']
    bem_fullpath = filepath['bem']
    return mne.make_forward_solution(
                preproc_fullpath,
                trans=trans_fullpath,
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                mindist=5.0,
                n_jobs=-1,
                verbose=True,
            )

def get_noise_cov(filepath):
    noise_fullpath = filepath['noise']
    noise_raw = mne.io.read_raw_ctf(noise_fullpath, preload=True)
    return mne.compute_raw_covariance(
                noise_raw, method=["shrunk", "empirical"], rank=None, verbose=True
            )

def get_inverse_solution(filepath, fwd, noise_cov):
    epoch = mne.read_epochs(filepath['epoch'], preload=True)
    info = epoch.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
    return apply_inverse(
                epoch,
                inverse_operator,
                lambda2=invsol_params['lambda2'],
                method=invsol_params['method'],
                pick_ori=None,
                return_residual=False,
                verbose=True,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    bloc = args.run
    
    filepath = create_fnames(subject, bloc)

    if not os.path.exists(filepath['fwd']):
        src = get_source_space(subject)
        bem = get_bem(subject)
        fwd = get_forward(filepath, src, bem)
        fwd.save(filepath['fwd'])
    else:
        fwd = mne.read_forward_solution(filepath['fwd'])

    if not os.path.exists(filepath['noise_cov']):
        noise_cov = get_noise_cov(filepath)
        noise_cov.save(filepath['noise_cov'])
    else:
        noise_cov = mne.read_cov(filepath['noise_cov'])    

    stc, residual = get_inverse_solution(filepath, fwd, noise_cov)
    
    # Save sources as hdf5
    stc.save(filepath['stc'])
    # Save residual as json
    residual_fullpath = filepath['stc'].replace('source', 'residual').replace('.h5', '.json')
    with open(residual_fullpath, 'w') as f:
        json.dump(residual, f)
        
