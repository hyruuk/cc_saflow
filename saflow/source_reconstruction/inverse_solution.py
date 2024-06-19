# Computes inverse solution and saves sources
from saflow import FS_SUBJDIR, SUBJ_LIST, BLOCS_LIST, BIDS_PATH
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw, apply_inverse_epochs
import os
import os.path as op
from mne_bids import BIDSPath, read_raw_bids
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='12',
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
    raw = read_raw_bids(raw_bidspath)
    info = raw.info
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
    bem_bidspath.mkdir(exist_ok=True)
    
    # Coregistration transform
    trans_bidspath = BIDSPath(subject=subject, 
                            task='gradCPT', 
                            run=bloc, 
                            datatype='meg', 
                            processing='trans',
                            root=BIDS_PATH + '/derivatives/trans/')
    trans_bidspath.mkdir(exist_ok=True)

    # Forward solution
    fwd_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='forward',
                            root=BIDS_PATH + '/derivatives/fwd/')
    fwd_bidspath.mkdir(exist_ok=True)

    # Noise covariance matrix
    noise_cov_bidspath = BIDSPath(subject='emptyroom', 
                    session=er_date,
                    task='noise', 
                    datatype="meg",
                    root=BIDS_PATH + '/derivatives/noise_cov/')
    noise_cov_bidspath.mkdir(exist_ok=True)

    # Sources
    stc_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='clean',
                            description='sources',
                            root=BIDS_PATH + '/derivatives/minimum-norm-estimate/')
    #stc_bidspath.mkdir(exist_ok=True)

    morph_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing='clean',
                            description='morphed',
                            root=BIDS_PATH + '/derivatives/morphed_sources/')
    morph_bidspath.mkdir(exist_ok=True)
    

    return {'raw':raw_bidspath,
            'preproc':preproc_bidspath,
            'epoch':epoch_bidspath,
            'bem':bem_bidspath,
            'noise':noise_bidspath,
            'trans':trans_bidspath,
            'fwd':fwd_bidspath,
            'noise_cov':noise_cov_bidspath,
            'stc':stc_bidspath,
            'morph':morph_bidspath}

def get_coregistration(filepath, subject, subjects_dir=FS_SUBJDIR, mri_available=False):
    raw = read_raw_bids(filepath['raw'])
    info = raw.info
    if mri_available:
        subject = 'sub-' + str(subject)
    else:
        subject = 'fsaverage'
    coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials="auto")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    os.makedirs(op.dirname(filepath['trans']), exist_ok=True)
    mne.write_trans(str(filepath['trans'].fpath)+'.fif', coreg.trans, overwrite=True)

    return coreg

def get_source_space(subject, subjects_dir=FS_SUBJDIR, mri_available=False):
    # Compute Source Space
    if mri_available:
        return mne.setup_source_space(
                    'sub-' + str(subject), spacing="oct6", add_dist="patch", subjects_dir=subjects_dir
                )
    else:
        return mne.setup_source_space(
                    'fsaverage', spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)

def get_bem(subject, subjects_dir=FS_SUBJDIR, mri_available=False):
    if mri_available:
        subject = 'sub-' + str(subject)
    else:
        subject = 'fsaverage'
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers (EEG)
    model = mne.make_bem_model(
        subject=subject, ico=5, conductivity=conductivity, subjects_dir=subjects_dir
    )
    return mne.make_bem_solution(model)

def get_forward(filepath, src, bem):
    return mne.make_forward_solution(
                str(filepath['preproc'].fpath),
                trans=str(filepath['trans'].fpath),
                src=src,
                bem=bem,
                meg=True,
                eeg=False,
                mindist=0.0,
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
    stcs = apply_inverse_epochs(
                epoch,
                inverse_operator,
                lambda2=lambda2,
                method=method,
                pick_ori=None,
                verbose=True,
            )
    # Save sources as hdf5
    for idx, stc in enumerate(stcs):
        filename = str(filepath['stc'].fpath).replace('clean', 'epo') + f'_epoch{idx}'
        stc.save(filename, ftype='h5', overwrite=True)
    # Save residual as json
    #residual_fullpath = str(filepath['stc'].fpath).replace('clean', 'epo').replace('source', 'residual').replace('.h5', '.json')
    #with open(residual_fullpath, 'w') as f:
    #    json.dump(residual, f)
    return stcs
    

def get_inverse(filepath, fwd, noise_cov):
    preproc = mne.io.read_raw_fif(filepath['preproc'], preload=True)
    info = preproc.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
    stc =  apply_inverse_raw(
                preproc,
                inverse_operator,
                lambda2=lambda2,
                method=method,
                pick_ori=None,
                verbose=True,
            )

    return stc

def get_morphed(filepath, subject, stcs, fwd, mri_available=False, subjects_dir=FS_SUBJDIR):
    # TODO : modify the function so it only accepts continuous signal
    fsaverage_fpath = op.join(FS_SUBJDIR, 'fsaverage', 'bem', 'fsaverage-oct-6-src.fif')
    # Create source space to project to
    src_to = get_source_space(subject, mri_available=False)
    if not mri_available:
        subject = 'fsaverage'
    else:
        subject = 'sub-' + str(subject)
    src_to = mne.read_source_spaces(fsaverage_fpath)
    morphed = []
    if len(stcs) > 1:
        fname_mrp = filepath['morph'].update(processing='epo')
    else:
        fname_mrp = filepath['morph'].update(processing='clean')
    # Morph each source estimate and save
    for idx, stc in enumerate(stcs):
        morph = mne.compute_source_morph(
            fwd['src'],
            subject_from=subject,
            src_to=src_to,
            subject_to="fsaverage",
            subjects_dir=subjects_dir,
        ).apply(stc)
        fname_mrp = filepath['morph']
        if len(stcs) > 1:
            filename = str(fname_mrp.update(processing='epo').fpath) + f'_epoch{idx}'
        else:
            filename = str(fname_mrp.update(processing='clean').fpath)

        stc_to_save = mne.SourceEstimate(data=np.float32(morph.data), 
                                        vertices=morph.vertices, 
                                        tmin=morph.tmin, 
                                        tstep=morph.tstep, 
                                        subject=subject)
        stc_to_save.save(filename, ftype='h5', overwrite=True)

    return stc_to_save


if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    bloc = args.run

    # Set some params
    method = "MNE"
    snr = 3.0
    lambda2 = 1.0 / snr**2

    # Check if subject in FS_SUBJDIR
    if not os.path.exists(FS_SUBJDIR + '/sub-' + str(subject)):
        mri_available = False
    else:
        mri_available = True
    print(f"MRI available : {mri_available}")

    # Create filenames
    filepath = create_fnames(subject, bloc)
    print(filepath)

    # Start processing
    if not os.path.exists(filepath['trans']):
        coreg = get_coregistration(filepath, subject, mri_available=mri_available)

    src = get_source_space(subject, mri_available=mri_available)

    if not os.path.exists(filepath['fwd']):
        bem = get_bem(subject, mri_available=mri_available)
        fwd = get_forward(filepath, src, bem)
        mne.write_forward_solution(str(filepath['fwd'].fpath) + '.fif', fwd, overwrite=True)
    else:
        fwd = mne.read_forward_solution(filepath['fwd'])

    if not os.path.exists(filepath['noise_cov']):
        noise_cov = get_noise_cov(filepath)
        os.makedirs(os.path.dirname(filepath['noise_cov']), exist_ok=True)
        noise_cov.save(filepath['noise_cov'])
    else:
        noise_cov = mne.read_cov(filepath['noise_cov'])

    # Apply inverse on preprocessed data
    stc = get_inverse(filepath, fwd, noise_cov)

    # Apply inverse on epochs
    #stcs = get_inverse_epochs(filepath, fwd, noise_cov)
        
    # Morph to fsaverage
    #if mri_available:
    stc_to_save = get_morphed(filepath, subject, [stc], fwd, mri_available=mri_available)
        #get_morphed(filepath, subject, stcs, src)
    #else:
        #stc.save(str(filepath['morph'].update(processing='clean')), ftype='h5', overwrite=True)
        #for idx, stc in enumerate(stcs):
            #stc.save(str(filepath['morph'].update(processing='epo').fpath), ftype='h5', overwrite=True)