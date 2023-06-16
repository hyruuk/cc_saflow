import mne
from mne_bids import write_raw_bids, BIDSPath
from heudiconv.cli.run import main as heudiconv_main
import scipy.io as sio
import os
import glob
import json

# MEG
ds_folders = glob.glob('/path/to/your/data/*')  # adjust this to your needs

for ds_folder in ds_folders:
    ds_files = glob.glob(os.path.join(ds_folder, '*.ds'))
    
    for ds_file in ds_files:
        raw = mne.io.read_raw_ctf(ds_file)

        # determine subject ID, run and task info from the file name
        file_name = os.path.basename(ds_file)
        if 'NOISE' in file_name:
            continue

        subject_id = file_name.split('SA')[1][:2]
        run = file_name.split('.')[0][-2:]

        # create a BIDS path
        bids_path = BIDSPath(subject=subject_id, run=run,
                             task='task' if run in ['02', '03', '04', '05', '06', '07'] else 'rest',
                             root='/path/to/your/bids/root',
                             datatype='meg')
        write_raw_bids(raw, bids_path, event_id=None if run in ['01', '08'] else event_id_dict)

# MRI
heudiconv_main(['-d', '/path/to/your/data/{subject}/*/*.dcm',
                '-s', 'SAflow_*',
                '-f', 'convertall',
                '-c', 'dcm2niix',
                '-b',
                '-o', '/path/to/your/bids/root'])

# Behavioral
mat_files = glob.glob('/path/to/your/data/behavioral/*.mat')

for mat_file in mat_files:
    data = sio.loadmat(mat_file)
    
    file_name = os.path.basename(mat_file).split('_')
    subject_id = file_name[2]
    run = file_name[3].zfill(2)

    # write events.tsv files for the task runs
    if run in ['02', '03', '04', '05', '06', '07']:
        events = data['events']  # adjust this to your needs
        bids_path = BIDSPath(subject=subject_id, run=run,
                             task='task',
                             root='/path/to/your/bids/root',
                             datatype='meg',
                             suffix='events',
                             extension='.tsv')
        events.to_csv(bids_path, sep='\t', index=False)