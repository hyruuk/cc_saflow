# Computes inverse solution and saves sources
from saflow import FS_SUBJDIR, SUBJ_LIST, BLOCS_LIST, BIDS_PATH
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
import os
from mne_bids import BIDSPath
import json



for subject in SUBJ_LIST:
    for bloc in BLOCS_LIST:
        bloc = '0'+bloc

        # Setup input files
        preproc_bidspath = BIDSPath(subject=subject, 
                                task='gradCPT', 
                                run=bloc, 
                                datatype='meg', 
                                suffix='meg',
                                processing='clean',
                                root=BIDS_PATH + '/derivatives/preprocessed/')
        preproc_fullpath = str(preproc_bidspath.fpath) + '.fif'

        trans_bidspath = BIDSPath(subject=subject, 
                                task='gradCPT', 
                                run=bloc, 
                                datatype='meg', 
                                processing='trans',
                                root=BIDS_PATH + '/derivatives/trans/')
        trans_fullpath = str(trans_bidspath.fpath) + '.fif'

        epochs_bidspath = BIDSPath(subject=subject, 
                                task='gradCPT', 
                                run=bloc, 
                                datatype='meg', 
                                processing='epo',
                                root=BIDS_PATH + '/derivatives/epochs/')
        epochs_fullpath = str(epochs_bidspath.fpath) + '.fif'

        # Find date
        info = mne.io.read_info(preproc_fullpath)
        er_date = info['meas_date'].strftime('%Y%m%d')
        # Get noise covariance matrix
        noise_cov_bidspath = BIDSPath(subject='emptyroom', 
                            session=er_date,
                            task='noise', 
                            datatype="meg",
                            root=BIDS_PATH)
        noise_cov_fullpath = str(noise_cov_bidspath.fpath)
        
        # Compute Source Space
        src = mne.setup_source_space(
            subject, spacing="oct6", add_dist="patch", subjects_dir=FS_SUBJDIR
        )

        conductivity = (0.3,)  # for single layer
        # conductivity = (0.3, 0.006, 0.3)  # for three layers (EEG)
        model = mne.make_bem_model(
            subject=subject, ico=4, conductivity=conductivity, subjects_dir=FS_SUBJDIR
        )
        bem = mne.make_bem_solution(model)

        # Compute forward operator (leadfield matrix)
        fwd = mne.make_forward_solution(
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

        # Save forward operator
        fwd_bidspath = BIDSPath(subject=subject,
                                task='gradCPT',
                                run=bloc,
                                datatype='meg',
                                processing='forward',
                                root=BIDS_PATH + '/derivatives/fwd/')
        
        fwd_fullpath = str(fwd_bidspath.fpath) + '.fif'
        os.makedirs(os.path.dirname(fwd_fullpath), exist_ok=True)
        mne.write_forward_solution(fwd_fullpath, fwd, overwrite=True)

        # Compute noise covariance matrix
        noise_raw = mne.io.read_raw_ctf(noise_cov_fullpath, preload=True)
        noise_cov = mne.compute_raw_covariance(
            noise_raw, method=["shrunk", "empirical"], rank=None, verbose=True
        )

        # Make inverse operator
        inverse_operator = make_inverse_operator(
            info, fwd, noise_cov, loose=0.2, depth=0.8
        )

        # Load epochs
        epoch = mne.read_epochs(epochs_fullpath, preload=True)
        # Compute inverse solution
        method = "MNE"
        snr = 3.0
        lambda2 = 1.0 / snr**2
        stc, residual = apply_inverse(
            epoch,
            inverse_operator,
            lambda2,
            method=method,
            pick_ori=None,
            return_residual=True,
            verbose=True,
        )

        # Save stc
        stc_bidspath = BIDSPath(subject=subject,
                                task='gradCPT',
                                run=bloc,
                                datatype='meg',
                                processing='source',
                                root=BIDS_PATH + '/derivatives/minimum-norm-estimate/')
        stc_fullpath = str(stc_bidspath.fpath) + '.h5'
        stc.save(stc_fullpath)

        # Save residual as json
        residual_fullpath = stc_fullpath.replace('source', 'residual').replace('.h5', '.json')
        with open(residual_fullpath, 'w') as f:
            json.dump(residual, f)
        
