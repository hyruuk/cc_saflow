import mne_bids
import mne
import os
import os.path as op
import numpy as np
from mne_bids import make_bids_folders, make_bids_basename, write_raw_bids
from src.saflow_params import BIDS_PATH, ACQ_PATH

# Define constants
EVENT_ID = {'Freq': 21, 'Rare': 31, 'Response': 99}

# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print('BIDS folder created at : {}'.format(BIDS_PATH))
else:
    print('{} already exists.'.format(BIDS_PATH))

# list folders in acquisition folder
recording_folders = os.listdir(ACQ_PATH)

# loop across recording folders (folder containing the recordings of the day)
for rec_date in recording_folders: # folders are named by date in format YYYYMMDD
    filelist = os.listdir(op.join(ACQ_PATH, rec_date))
    subjects_in_folder = np.unique([filename[2:4] for filename in filelist if 'SA' in filename])
    for file in filelist:
        # Create emptyroom BIDS if doesn't exist already
        if 'NOISE_noise' in file:
            for sub in subjects_in_folder:
                noise_basename = 'sub-{}_ses-recording_NOISE'.format(sub)
                if not op.isdir(op.join(BIDS_PATH, 'sub-{}'.format(sub), 'ses-recording', 'meg', noise_basename)):
                    er_bids_basename = make_bids_basename(subject='emptyroom', session=rec_date)
                    er_raw_fname = op.join(ACQ_PATH, rec_date, file)
                    er_raw = mne.io.read_raw_ctf(er_raw_fname)
                    write_raw_bids(er_raw, noise_basename, BIDS_PATH)
        # Rewrite in BIDS format if doesn't exist yet
        if 'SA' in file and '.ds' in file and not 'procedure' in file:
            subject = file[2:4]
            run = file[-5:-3]
            session = 'recording'
            if run == '01' or run == '08':
                task = 'RS'
            else:
                task = 'gradCPT'
            bids_basename = make_bids_basename(subject=subject, session=session, task=task, run=run)
            if not op.isdir(op.join(BIDS_PATH, 'sub-{}'.format(subject), 'ses-{}'.format(session), 'meg', bids_basename + '_meg.ds')):
                raw_fname = op.join(ACQ_PATH, rec_date, file)
                raw = mne.io.read_raw_ctf(raw_fname, preload=False)
                try:
                    events = mne.find_events(raw)
                    write_raw_bids(raw, bids_basename, BIDS_PATH, events_data=events, event_id=EVENT_ID, overwrite=True)
                except:
                    write_raw_bids(raw, bids_basename, BIDS_PATH, overwrite=True)
