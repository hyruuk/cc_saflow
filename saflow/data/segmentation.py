##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow.utils import get_SAflow_bids
from saflow.neuro import segment_files
from saflow import BIDS_PATH, LOGS_DIR, SUBJ_LIST, BLOCS_LIST
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default="04",
    type=str,
    help="Subject to process",
)
args = parser.parse_args()

if __name__ == "__main__":
    subj = args.subject
    for bloc in BLOCS_LIST:
        preproc_path, preproc_filename = get_SAflow_bids(
            BIDS_PATH, subj, bloc, stage="preproc_raw", cond=None
        )
        epoched_path, epoched_filename = get_SAflow_bids(
            BIDS_PATH, subj, bloc, stage="-epo4001200", cond=None
        )
        ARlog_path, ARlog_filename = get_SAflow_bids(
            BIDS_PATH, subj, bloc, stage="ARlog4001200", cond=None
        )
        if not os.path.isfile(epoched_filename):
            epochs_clean, AR_log = segment_files(preproc_filename, tmin=0.4, tmax=1.2)
            epochs_clean.save(epoched_filename, overwrite=True)
            del epochs_clean
            with open(ARlog_filename, "wb") as fp:
                pickle.dump(AR_log, fp)
        else:
            print("{} {} File already exists".format(subj, bloc))
