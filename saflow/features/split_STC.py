##### OPEN PREPROC FILES AND SEGMENT THEM
from ephypype.import_data import write_hdf5
from saflow.neuro import get_VTC_epochs
from saflow import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH, LOGS_DIR
import h5py

if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for bloc in BLOCS_LIST:
            try:
                sources_fp = FOLDERPATH + 'source_reconstruction_MNE_aparca2009s/inv_sol_pipeline/'
                template_path = sources_fp + '_run_id_run-0{}_session_id_ses-recording_subject_id_sub-{}/inv_solution/sub-{}_ses-recording_task-gradCPT_run-0{}_meg_-epo_stc.hdf5'.format(bloc, subj, subj, bloc)
                template_IN = sources_fp + '_run_id_run-0{}_session_id_ses-recording_subject_id_sub-{}/inv_solution/sub-{}_ses-recording_task-gradCPT_run-0{}_meg_IN_-epo_stc.hdf5'.format(bloc, subj, subj, bloc)
                template_OUT = sources_fp + '_run_id_run-0{}_session_id_ses-recording_subject_id_sub-{}/inv_solution/sub-{}_ses-recording_task-gradCPT_run-0{}_meg_OUT_-epo_stc.hdf5'.format(bloc, subj, subj, bloc)

                with h5py.File(template_path, "r") as f:
                    # List all groups
                    a_group_key = list(f.keys())[0]

                    # Get the data
                    data = list(f[a_group_key])

                INidx, OUTidx, VTC_epo, idx_trimmed = get_VTC_epochs(LOGS_DIR, subj, bloc, stage='-epo', lobound=None, hibound=None, save_epochs=False, filt_order=3, filt_cutoff=0.1)
                data_trimmed = [data[i] for i in idx_trimmed]
                data_IN = [data_trimmed[i] for i in INidx]
                data_OUT = [data_trimmed[i] for i in OUTidx]

                write_hdf5(template_IN, data_IN, dataset_name='stc_data', dtype='f')
                write_hdf5(template_OUT, data_OUT, dataset_name='stc_data', dtype='f')
                print('subj {} bloc {} splitted'.format(subj, bloc))
            except:
                print('file missing')
