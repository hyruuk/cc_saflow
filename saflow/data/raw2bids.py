import mne
from mne_bids import write_raw_bids, BIDSPath
#from heudiconv.cli.run import main as heudiconv_main
import scipy.io as sio
import os.path as op
import os
import glob
import json
import argparse
from mne import Annotations
import mne
from mne.datasets import sample
from typing import Tuple, List
import mne
import numpy as np
from mne_bids import (write_raw_bids, read_raw_bids,
                      BIDSPath, print_dir_tree)
import pandas as pd
from saflow.behav import get_VTC_from_file
from saflow import SUBJ_LIST

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


def parse_info_from_name(fname: str) -> Tuple[str, str, str]:
    """
    Parses subject, run, and task information from a given filename.

    Parameters
    ----------
    fname : str
        The filename to parse information from.

    Returns
    -------
    tuple
        A tuple containing the subject, run, and task information.
    """
    subject = fname.split("SA")[1][:2]
    run = fname.split("_")[-1][:2]
    if run in ["01", "08"]:
        task = "rest"
    else:
        task = "gradCPT"
    return subject, run, task


def write_noise_file(ds_file: str, bids_root: str) -> None:
    """
    Writes a noise file in BIDS format.

    Parameters
    ----------
    ds_file : str
        The path to the noise file.
    bids_root : str
        The path to the BIDS root directory.
    """
    er_raw = mne.io.read_raw_ctf(ds_file)
    er_raw.info['line_freq'] = 60
    er_date = er_raw.info['meas_date'].strftime('%Y%m%d')
    er_bids_path = BIDSPath(subject='emptyroom', 
                            session=er_date,
                            task='noise', 
                            datatype="meg",
                            extension=".fif",
                            root=bids_root)
    write_raw_bids(er_raw, er_bids_path, format="FIF", overwrite=True)


def load_recording(fname: str, bids_root: str) -> Tuple[mne.io.Raw, BIDSPath, str]:
    """
    Loads a recording and returns the MNE raw object, BIDS path, and task information.

    Parameters
    ----------
    fname : str
        The path to the recording file.
    bids_root : str
        The path to the BIDS root directory.

    Returns
    -------
    tuple
        A tuple containing the MNE raw object, BIDS path, and task information.
    """
    subject, run, task = parse_info_from_name(fname)
    bidspath = BIDSPath(subject=subject,
                        task=task,
                        run=run,
                        datatype="meg",
                        extension=".fif",
                        root=bids_root)
    raw = mne.io.read_raw_ctf(ds_file)
    raw.info['line_freq'] = 60
    mne.rename_channels(raw.info,{'EEG057':'ECG', 
                                  'EEG058':'hEOG', 
                                  'EEG059': 'vEOG'})
    return raw, bidspath, task

def get_events(raw: mne.io.Raw):
    """
    Finds events in a given MNE raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The MNE raw object to find events in.

    Returns
    -------
    tuple
        A tuple containing the events found and their corresponding event IDs.
    """
    event_id = {"Freq": 21, "Rare": 31, "Resp": 99, "BlocStart": 10}
    #try:
    events_mne = mne.find_events(raw, verbose=False)
    #except ValueError:
    #    events_mne = mne.find_events(raw, min_duration=2 / raw.info["sfreq"], verbose=False)
    return events_mne, event_id

def add_trial_idx(events_bids: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a trial index column to a pandas DataFrame containing BIDS events.

    Parameters
    ----------
    events_bids : pd.DataFrame
        The pandas DataFrame containing BIDS events.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame with a new column containing trial indices.
    """
    current_trial_idx = 0
    trial_idx_list = []
    for event in events_bids.itertuples():
        if event.trial_type in ['Freq', 'Rare']:
            trial_idx_list.append(current_trial_idx)
            current_trial_idx += 1
        else:
            trial_idx_list.append(-1)
    events_bids['trial_idx'] = np.array(trial_idx_list).astype(int)
    return events_bids

def find_trial_in_dict(type_dict: dict, idx_tofind: int) -> str:
    """
    Finds the trial for a given trial index.

    Parameters
    ----------
    performance_dict : dict
        A dictionary containing the trial performance information.
    idx_tofind : int
        The index of the trial to find the performance for.

    Returns
    -------
    str
        The trial performance for the given trial index.
    """
    for trial_type, idx_list in type_dict.items():
        if idx_tofind in idx_list:
            return trial_type
        
def add_behav_info(events_bids: pd.DataFrame, VTC_raw: List[float], RT_to_VTC: List[float], performance_dict: dict) -> pd.DataFrame:
    """
    Adds behavioral information to a pandas DataFrame containing BIDS events.

    Parameters
    ----------
    events_bids : pd.DataFrame
        The pandas DataFrame containing BIDS events.
    VTC_raw : List[float]
        A list of VTC values.
    RT_to_VTC : List[float]
        A list of RT values.
    performance_dict : dict
        A dictionary containing the trial performance information.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame with new columns containing behavioral information.
    """
    VTC_list = []
    RT_list = []
    task_list = []
    for event in events_bids.itertuples():
        if event.trial_type in ['Freq', 'Rare']:
            trial_idx = int(event.trial_idx)
            VTC_list.append(VTC_raw[trial_idx])
            RT_list.append(RT_to_VTC[trial_idx])
            task_list.append(find_trial_in_dict(performance_dict, trial_idx))
        else:
            VTC_list.append('n/a')
            RT_list.append(0)
            task_list.append('n/a')
    events_bids['VTC'] = VTC_list
    events_bids['RT'] = RT_list
    events_bids['task'] = task_list
    return events_bids

def add_in_out_zone(events_bids: pd.DataFrame, bidspath: BIDSPath, files_list: List[str]) -> pd.DataFrame:
    """
    Adds a column to a pandas DataFrame containing BIDS events, indicating whether each trial is in or out of the zone.

    Parameters
    ----------
    events_bids : pd.DataFrame
        The pandas DataFrame containing BIDS events.
    bidspath : BIDSPath
        The BIDSPath object containing information about the BIDS dataset.
    files_list : List[str]
        A list of file names.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame with a new column indicating whether each trial is in or out of the zone.
    """
    for bounds in [[50,50], [25, 75], [10, 90]]:
        (IN_idx, 
        OUT_idx, 
        VTC_raw, 
        VTC_filtered, 
        IN_mask, 
        OUT_mask, 
        performance_dict, 
        df_response_out, 
        RT_to_VTC) = get_VTC_from_file(bidspath.subject,
                                        bidspath.run[1],
                                        files_list,
                                        cpt_blocs=[2, 3, 4, 5, 6, 7],
                                        inout_bounds=bounds,
                                        filt_cutoff=0.05,
                                        filt_type="gaussian",
                                        )
        inout_dict = {'IN':IN_idx, 'OUT':OUT_idx}
        inout_list = []
        for event in events_bids.itertuples():
            if event.trial_type in ['Freq', 'Rare']:
                inout_list.append(find_trial_in_dict(inout_dict, event.trial_idx))
            else:
                inout_list.append('n/a')
        events_bids[f'INOUT_{bounds[0]}_{bounds[1]}'] = inout_list
    return events_bids

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    raw_folder = args.input
    bids_root = args.output
    os.makedirs(bids_root, exist_ok=True)

    # List MEG files
    meg_folder = op.join(raw_folder, "meg")
    ds_files = glob.glob(op.join(meg_folder, "*", "*.ds"))
    for ds_file in sorted(ds_files):
        fname = ds_file.split("/")[-1]
        if fname.split('_')[0][2:] in SUBJ_LIST:
            # Handle noise files
            if "NOISE1Trial5min" in fname:
                write_noise_file(ds_file, bids_root)
            # Then recording files
            elif "SA" in fname and not "procedure" in fname:
                raw, bidspath, task = load_recording(fname, bids_root)

                if task == "gradCPT":
                    events_mne, event_id = get_events(raw)
                    raw.set_annotations(Annotations([], [], []))
                    write_raw_bids(
                        raw,
                        bidspath,
                        events=events_mne,
                        event_id=event_id,
                        overwrite=True,
                        format="FIF"
                    )
                    # Get bids events to enrich them
                    events_bids_fname = bidspath.copy().update(suffix='events', extension='.tsv')
                    events_bids = pd.read_csv(events_bids_fname, sep='\t')
                    files_list = os.listdir(op.join(raw_folder, "behav"))
                    (IN_idx, 
                    OUT_idx, 
                    VTC_raw, 
                    VTC_filtered, 
                    IN_mask, 
                    OUT_mask, 
                    performance_dict, 
                    df_response_out, 
                    RT_to_VTC) = get_VTC_from_file(bidspath.subject,
                                                    bidspath.run[1],
                                                    files_list,
                                                    cpt_blocs=[2, 3, 4, 5, 6, 7],
                                                    inout_bounds=[25, 75],
                                                    filt_cutoff=0.05,
                                                    filt_type="gaussian",
                                                    )
                    events_bids = add_trial_idx(events_bids)
                    events_bids = add_behav_info(events_bids, VTC_raw, RT_to_VTC, performance_dict)
                    events_bids = add_in_out_zone(events_bids, bidspath, files_list)
                    events_bids.to_csv(events_bids_fname, sep='\t', index=False)
                else:
                    write_raw_bids(raw, 
                                bidspath, 
                                format="FIF", 
                                overwrite=True)
