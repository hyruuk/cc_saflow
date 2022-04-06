import mne
import numpy as np
from saflow.utils import get_SAflow_bids
from saflow.neuro import compute_PSD_hilbert, compute_PSD
from saflow import BIDS_PATH, IMG_DIR, FREQS, FREQS_NAMES, SUBJ_LIST, BLOCS_LIST
from scipy.io import savemat
import pickle
import argparse
from saflow.behav import *
from saflow.neuro import *
from saflow import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="04",
    type=str,
    help="Subject to process",
)

args = parser.parse_args()

### OPEN SEGMENTED FILES AND COMPUTE PSDS
if __name__ == "__main__":
    subj = args.subject
    tmin = 0.4
    tmax = 1.2
    for bloc in BLOCS_LIST:
        # Generate filenames
        _, epopath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage="-epo", cond=None)
        _, rawpath = get_SAflow_bids(
            BIDS_PATH, subj, bloc, stage="preproc_raw", cond=None
        )
        _, ARpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage="ARlog", cond=None)
        _, PSDpath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage="PSD", cond=None)

        # Load files
        epochs = mne.read_epochs(epopath)
        raw = mne.io.read_raw_fif(rawpath, preload=True)
        with open(ARpath, "rb") as f:
            ARlog = pickle.load(f)


        for idx_freq, freq_bounds in enumerate(FREQS):
            low = freq_bounds[0]
            high = freq_bounds[1]
            # Filter continuous data
            data = raw.copy().filter(low, high)  # Here epochs is a raw file
            hilbert = data.apply_hilbert(envelope=True)

            # Segment them
            picks = mne.pick_types(
                raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
            )
            for inout_bounds in [[50, 50], [25, 75], [10, 90]]:
                (
                    INidx,
                    OUTidx,
                    VTC_raw,
                    VTC_filtered,
                    IN_mask,
                    OUT_mask,
                    performance_dict,
                    df_response_out,
                ) = get_VTC_from_file(
                    subj, bloc, os.listdir(LOGS_DIR), inout_bounds=inout_bounds
                )

                events = mne.find_events(
                    raw, min_duration=2 / raw.info["sfreq"], verbose=False
                )
                logfile = LOGS_DIR + find_logfile(subj, bloc, os.listdir(LOGS_DIR))
                events = annotate_events(logfile, events, inout_idx=[INidx, OUTidx])
                event_id = get_present_events(events)
                epochs = mne.Epochs(
                    hilbert,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    reject=None,
                    picks=picks,
                    preload=True,
                )
                epochs.drop(ARlog.bad_epochs)

                _, PSDpath = get_SAflow_bids(
                    BIDS_PATH,
                    subj,
                    bloc,
                    stage=f"-epoenv4001200_{FREQS_NAMES[idx_freq]}",
                    cond=f"{inout_bounds[0]}{inout_bounds[1]}",
                )
                epochs.save(PSDpath, overwrite=True)
