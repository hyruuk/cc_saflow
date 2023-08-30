# Computes inverse solution and saves sources
from saflow import FS_SUBJDIR, SUBJ_LIST, BLOCS_LIST, BIDS_PATH, invsol_params

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs
import os
import os.path as op
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
    raw_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            suffix='meg',
                            extension='.ds',
                            root=BIDS_PATH)
    
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
                            processing='clean',
                            description='sources',
                            root=BIDS_PATH + '/derivatives/minimum-norm-estimate/')
    

    # Morph
    morph_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='clean',
                            description='morphed',
                            root=BIDS_PATH + '/derivatives/minimum-norm-estimate/')

    return {'raw':str(raw_bidspath.fpath),
            'preproc':str(preproc_bidspath.fpath) + '.fif',
            'epoch':str(epoch_bidspath.fpath) + '.fif',
            'bem':str(bem_bidspath.fpath) + '.h5',
            'noise':str(noise_bidspath.fpath) + '.ds',
            'trans':str(trans_bidspath.fpath) + '.fif',
            'fwd':str(fwd_bidspath.fpath) + '.fif',
            'noise_cov':str(noise_cov_bidspath.fpath) + '.fif',
            'stc':str(stc_bidspath.fpath) + '.h5',
            'morph':str(morph_bidspath.fpath) + '.h5'}

def get_coregistration(filepath, subject, subjects_dir=FS_SUBJDIR):
    info = mne.io.read_info(filepath['raw'])
    coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials="estimated")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    os.makedirs(op.dirname(filepath['trans']), exist_ok=True)
    mne.write_trans(filepath['trans'], coreg.trans, overwrite=True)

    return coreg

def get_source_space(subject, subjects_dir=FS_SUBJDIR):
    # Compute Source Space
    return mne.setup_source_space(
                subject, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir
            )

def get_bem(subject):
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers (EEG)
    model = mne.make_bem_model(
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
    )
    return mne.make_bem_solution(model)

def get_forward(filepath, src, bem):
    return mne.make_forward_solution(
                filepath['preproc'],
                trans=filepath['trans'],
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

def get_inverse_epochs(filepath, fwd, noise_cov):
    epoch = mne.read_epochs(filepath['epoch'], preload=True)
    info = epoch.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
    stcs, residual = apply_inverse_epochs(
                epoch,
                inverse_operator,
                lambda2=invsol_params['lambda2'],
                method=invsol_params['method'],
                pick_ori=None,
                return_residual=False,
                verbose=True,
            )
    # Save sources as hdf5
    for idx, stc in enumerate(stcs):
        filename = filepath['stc'].replace('clean', 'epo').replace('.h5', '_{}.h5'.format(idx))
        stc.save(filename)
    # Save residual as json
    residual_fullpath = filepath['stc'].replace('clean', 'epo').replace('source', 'residual').replace('.h5', '.json')
    with open(residual_fullpath, 'w') as f:
        json.dump(residual, f)
    return stcs, residual
    

def get_inverse(filepath, fwd, noise_cov):
    preproc = mne.io.read_raw_fif(filepath['preproc'], preload=True)
    info = preproc.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
    stc, residual =  apply_inverse(
                preproc,
                inverse_operator,
                lambda2=invsol_params['lambda2'],
                method=invsol_params['method'],
                pick_ori=None,
                return_residual=False,
                verbose=True,
            )
    # Save sources as hdf5
    stc.save(filepath['stc'])
    # Save residual as json
    residual_fullpath = filepath['stc'].replace('source', 'residual').replace('.h5', '.json')
    with open(residual_fullpath, 'w') as f:
        json.dump(residual, f)
    return stc, residual

def get_morphed(filepath, stcs, subjects_dir=FS_SUBJDIR):
    fsaverage_fpath = ''
    src_to = mne.read_source_spaces(fsaverage_fpath)
    morphed = []
    # Morph each source estimate and save
    for idx, stc in enumerate(stcs):
        morph = mne.compute_source_morph(
            stc,
            subject_from=subject,
            subject_to="fsaverage",
            src_to=src_to,
            subjects_dir=subjects_dir,
        ).apply(stc)
        filename = os.path.join(fname_mrp, f'{idx}_{method}_morph')
        morph.save(filename)
        morphed.append(morph)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    bloc = args.run
    
    filepath = create_fnames(subject, bloc)
    print(filepath)

    if not os.path.exists(filepath['trans']):
        coreg = get_coregistration(filepath)

    src = get_source_space(subject)

    if not os.path.exists(filepath['fwd']):
        bem = get_bem(subject)
        fwd = get_forward(filepath, src, bem)
        os.makedirs(os.path.dirname(filepath['fwd']), exist_ok=True)
        fwd.save(filepath['fwd'])
    else:
        fwd = mne.read_forward_solution(filepath['fwd'])

    if not os.path.exists(filepath['noise_cov']):
        noise_cov = get_noise_cov(filepath)
        os.makedirs(os.path.dirname(filepath['noise_cov']), exist_ok=True)
        noise_cov.save(filepath['noise_cov'])
    else:
        noise_cov = mne.read_cov(filepath['noise_cov'])    

    # Apply inverse on preprocessed data (without AR #TODO: add AR -> from segmentation script)
    stc, residual = get_inverse(filepath, fwd, noise_cov)

    # Apply inverse on epochs
    stcs, residual = get_inverse_epochs(filepath, fwd, noise_cov)
        
    # Morph to fsaverage
    morphed_preproc = get_morphed(filepath, [stc], src)[0]
    morphed_epoch = get_morphed(filepath, stcs, src)
