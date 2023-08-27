##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow.utils import get_SAflow_bids
from saflow.neuro import segment_files
from saflow import BIDS_PATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST
import pickle
import os
import argparse
from mne_bids import BIDSPath
from mne.io import read_raw_fif
import mne
from autoreject import AutoReject
from mne_bids import write_raw_bids, read_raw_bids

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="04",
    type=str,
    help="Subject to process",
)
args = parser.parse_args()

def segment_files(bids_filepath, tmin=0, tmax=0.8):
    raw = read_raw_fif(bids_filepath, preload=True)
    picks = mne.pick_types(
        raw.info, meg=True, ref_meg=True, eeg=False, eog=False, stim=False
    )
    ### Set some constants for epoching
    baseline = None  # (None, -0.05)
    # reject = {'mag': 4e-12}
    try:
        events = mne.find_events(raw, min_duration=1 / raw.info["sfreq"], verbose=False)
    except ValueError:
        events = mne.find_events(raw, min_duration=2 / raw.info["sfreq"], verbose=False)
    event_id = {"Freq": 21, "Rare": 31}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=None,
        picks=picks,
        preload=True,
    )
    #epochs = epochs[:10] # just for debugging, to remove
    ar = AutoReject(n_jobs=24)
    epochs_clean, autoreject_log = ar.fit_transform(epochs, return_log=True)
    return epochs_clean, autoreject_log

if __name__ == "__main__":
    subj = args.subject
    for bloc in BLOCS_LIST:
        input_file = BIDSPath(
            subject=subj,
            task="gradCPT",
            run="0" + bloc,
            datatype="meg",
            root=BIDS_PATH + '/derivatives/preprocessed/',
            processing='clean', 
            suffix='meg'
        )
        epoch_file = BIDSPath(
            subject=subj,
            task="gradCPT",
            run="0" + bloc,
            datatype="meg",
            root=BIDS_PATH + '/derivatives/epochs/',
            processing='epo',
            suffix='meg',
            extension='.fif'
        )
        #epoch_file = str(input_file.copy().update(processing='epo'))
        #ARlog_file = str(input_file.copy().update(processing='epo', description='ARlog')).replace('.fif', '.pkl')

        #if not os.path.isfile(epoch_file):
        epochs_clean, AR_log = segment_files(input_file, tmin=0.512, tmax=1.532)
        write_raw_bids(read_raw_bids(input_file), epoch_file)
        epoch_file = str(epoch_file.fpath)
        ARlog_file = epoch_file.replace('meg.fif', 'ARlog.pkl')
        epochs_clean.save(epoch_file, overwrite=True)
        del epochs_clean
        print(ARlog_file)
        with open(ARlog_file, "wb") as fp:
            pickle.dump(AR_log, fp)
            
        #else:
        #    print("{} {} File already exists".format(subj, bloc))