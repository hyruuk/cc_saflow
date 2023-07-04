import mne
from mne_bids import write_raw_bids, BIDSPath
from heudiconv.cli.run import main as heudiconv_main
import scipy.io as sio
import os.path as op
import os
import glob
import json
import argparse
from datetime import datetime, timezone

import mne
from mne.datasets import sample

from mne_bids import (write_raw_bids, read_raw_bids,
                      BIDSPath, print_dir_tree)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    default="/media/hyruuk/CoCoLabYANN/coco_data/saflow/raw",#'./data/raw',
    type=str,
    help="Path to the raw data folder",
)
parser.add_argument(
	"-o",
	"--output",
	default = "/media/hyruuk/CoCoLabYANN/coco_data/saflow/bids",#"./data/bids",
	type = str,
	help="Path to the output BIDS folder"
)

# Define constants
EVENT_ID = {"Freq": 21, "Rare": 31, "Response": 99, "BlocStart": 10}

def parse_info_from_name(fname):
    # Parse info from a recording file
    subject = fname.split("SA")[1][:2]
    run = fname.split("_")[-1][:2]
    if run in ["01", "07"]:
        task = "rest"
    else:
        task = "gradCPT"
    return subject, run, task

def write_noise_file(ds_file, bids_root):
    er_raw = mne.io.read_raw_ctf(ds_file)
    er_raw.info['line_freq'] = 60
    er_date = er_raw.info['meas_date'].strftime('%Y%m%d')
    er_bids_path = BIDSPath(subject='emptyroom', 
                            session=er_date,
                            task='noise', 
                            datatype="meg",
                            extension=".fif",
                            root=bids_root)
    write_raw_bids(er_raw, er_bids_path, overwrite=False)

def load_recording(fname, bids_root):
    subject, run, task = parse_info_from_name(fname)
    bidspath = BIDSPath(subject=subject,
                        task=task,
                        run=run,
                        datatype="meg",
                        extension=".fif",
                        root=bids_root)
    raw = mne.io.read_raw_ctf(ds_file)
    raw.info['line_freq'] = 60
    return raw, bidspath, task

def events_from_file(raw):
    events_mne, event_id = mne.events_from_annotations(raw)
    event_id_inverted = {v: k for k, v in event_id.items()}
    sampling_rate = raw.info['sfreq']
    events_bids = {'onset': events_mne[:, 0] / sampling_rate,
                   'duration': events_mne[:, 1] / sampling_rate,
                   'trial_type': [event_id_inverted[i] for i in events_mne[:, 2]]}
    return events_mne, events_bids, event_id

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    raw_folder = args.input
    bids_root = args.output
    os.makedirs(bids_root, exist_ok=True)
    # List MEG files
    meg_folder = op.join(raw_folder, "meg")
    ds_folders = os.listdir(meg_folder)

    ds_files = glob.glob(op.join(meg_folder, "*", "*.ds"))
    for ds_file in sorted(ds_files):
        fname = ds_file.split("/")[-1]
        # Handle noise files
        if "NOISE1Trial5min" in fname:
            write_noise_file(ds_file, bids_root)
        # Then recording files
        elif "SA" in fname and not "procedure" in fname:
            raw, bidspath, task = load_recording(fname, bids_root)
            if task == "gradCPT":
                events_mne, events_bids, event_id = events_from_file(raw)
                0/0
                write_raw_bids(
                    raw,
                    bidspath,
                    events_data=events,
                    event_id=event_id,
                    overwrite=True,
                )
                
            else:
                write_raw_bids(raw, bidspath, overwrite=False)

            



'''
# MEG
ds_folders = glob.glob(op.join(raw_folder, "meg"))  # adjust this to your needs

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
'''

