from saflow_params import FS_SUBJDIR, FOLDERPATH, SUBJ_LIST, BLOCS_LIST
import numpy as np
import os
import os.path as op
import mne
import scipy.io as sio
import h5py
from ephypype.import_data import write_hdf5

fwd_template = FOLDERPATH + '/sub-{subj}/ses-recording/meg/sub-{subj}_ses-recording_task-gradCPT_run-0{bloc}_meg_-epo-oct-6-fwd.fif'
sources_fp = FOLDERPATH
sources_template = FOLDERPATH + 'source_reconstruction_MNE_aparca2009s/inv_sol_pipeline/_run_id_run-0{bloc}_session_id_ses-recording_subject_id_sub-{subj}/inv_solution/sub-{subj}_ses-recording_task-gradCPT_run-0{bloc}_meg_-epo_stc.hdf5'
morphed_template = FOLDERPATH + 'source_reconstruction_MNE_aparca2009s/inv_sol_pipeline/_run_id_run-0{bloc}_session_id_ses-recording_subject_id_sub-{subj}/inv_solution/sub-{subj}_ses-recording_task-gradCPT_run-0{bloc}_meg_-epo_stcmorphed.hdf5'


fsaverage_fpath = op.join(FS_SUBJDIR, 'fsaverage/bem/fsaverage-oct-6-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_fpath)
vertices_to = [s['vertno'] for s in fsaverage_src]

for subj in SUBJ_LIST:
    for bloc in BLOCS_LIST:
        fwd_fpath = fwd_template.format(subj=subj, bloc=bloc)
        sources_fpath = sources_template.format(subj=subj, bloc=bloc)
        morphed_fpath = morphed_template.format(subj=subj, bloc=bloc)

        fwd = mne.read_forward_solution(fwd_fpath)
        src = fwd['src']
        surf_src = mne.source_space.SourceSpaces(fwd['src'][:2])
        n_cortex = (src[0]['nuse'] + src[1]['nuse'])
        try:
            morph_surf = mne.compute_source_morph(
                src=surf_src, subject_from='sub-{}'.format(subj), subject_to='fsaverage',
                spacing=vertices_to, subjects_dir=FS_SUBJDIR)
        except ValueError:
            try:
                morph_surf = mne.compute_source_morph(
                    src=surf_src, subject_from='SA{}'.format(subj), subject_to='fsaverage',
                    spacing=vertices_to, subjects_dir=FS_SUBJDIR)
            except ValueError:
                morph_surf = mne.compute_source_morph(
                    src=surf_src, subject_from='fsaverage', subject_to='fsaverage',
                    spacing=vertices_to, subjects_dir=FS_SUBJDIR)

        with h5py.File(sources_fpath, 'r') as f:
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])

        morphed_data = []
        morphed_dat = np.array([])
        for i, dat in enumerate(data):
            morphed_data.append(morphed_dat)
            morphed_data[i] = morph_surf.morph_mat*dat

        write_hdf5(morphed_fpath, morphed_data, dataset_name='stc_data', dtype='f')
        del data, morphed_data
