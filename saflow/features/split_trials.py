##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow.neuro import split_PSD_data, split_trials
from saflow import SUBJ_LIST, BLOCS_LIST, FEAT_PATH, BIDS_PATH, LOGS_DIR
from saflow.utils import get_SAflow_bids
from scipy.io import savemat
import pickle
import argparse
from saflow import *
import mne
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--split",
    default=[50, 50],
    type=int,
    nargs="+",
    help="Bounds of percentile split",
)
parser.add_argument(
    "-by",
    "--by",
    default="odd",
    type=str,
    help="Choose the basis on which to split the data ('VTC' or 'odd')",
)
parser.add_argument(
    "-i",
    "--input_stage",
    default="PSD4001200",
    type=str,
    help="Choose the PSD file to load (can be PSD, or PSD4001200 for epochs"
    "splitted between 400 and 1200ms)",
)


args = parser.parse_args()


def new_split_trials(subj, run, by="VTC", inout_bounds=None):
    condA = []
    condB = []
    for idx_freq, freq_bounds in enumerate(FREQS):
        _, PSDpath = get_SAflow_bids(
            BIDS_PATH, subj, run, stage=f"-epoenv4001200_{FREQS_NAMES[idx_freq]}", cond=f"{inout_bounds[0]}{inout_bounds[1]}"
        )

        epochs = mne.read_epochs(PSDpath)
        if by == "VTC":
            condA_epochs = epochs["FreqIN"]
            condB_epochs = epochs["FreqOUT"]
        elif by == "odd":
            condA_epochs = epochs[["FreqIN", "FreqOUT", "FreqHit"]]
            condB_epochs = epochs["RareHit"]
        condA.append(condA_epochs.get_data())
        condB.append(condB_epochs.get_data())
    condA = np.mean(np.array(condA), axis=3).transpose(1, 2, 0)
    condB = np.mean(np.array(condB), axis=3).transpose(1, 2, 0)
    return condA, condB


if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for run in BLOCS_LIST:
            CONDS_LIST = args.split
            by = args.by
            stage = args.input_stage

            if by == "VTC":
                INepochs, OUTepochs = new_split_trials(
                    subj=subj,
                    run=run,
                    by="VTC",
                    inout_bounds=CONDS_LIST
                )
                VTCepochs_path, VTCepochs_filename = get_SAflow_bids(
                    BIDS_PATH,
                    subj=subj,
                    run=run,
                    stage=stage,
                    cond=f"{CONDS_LIST[0]}{CONDS_LIST[1]}",
                )
                INepochs_path, INepochs_filename = get_SAflow_bids(
                    BIDS_PATH,
                    subj=subj,
                    run=run,
                    stage=stage,
                    cond="IN{}".format(CONDS_LIST[0]),
                )
                OUTepochs_path, OUTepochs_filename = get_SAflow_bids(
                    BIDS_PATH,
                    subj=subj,
                    run=run,
                    stage=stage,
                    cond="OUT{}".format(CONDS_LIST[1]),
                )

                with open(INepochs_filename, "wb") as fp:
                    pickle.dump(INepochs, fp)
                with open(OUTepochs_filename, "wb") as fp:
                    pickle.dump(OUTepochs, fp)
                with open(VTCepochs_filename, "wb") as fp:
                    pickle.dump([INepochs, OUTepochs], fp)

            elif by == "odd":
                FREQepochs, RAREepochs = new_split_trials(
                    subj=subj,
                    run=run,
                    by="odd",
                )
                FREQepochs_path, FREQepochs_filename = get_SAflow_bids(
                    BIDS_PATH, subj=subj, run=run, stage=stage, cond="FREQhits"
                )
                RAREepochs_path, RAREepochs_filename = get_SAflow_bids(
                    BIDS_PATH, subj=subj, run=run, stage=stage, cond="RAREhits"
                )

                ODDepochs_path, ODDepochs_filename = get_SAflow_bids(
                    BIDS_PATH, subj=subj, run=run, stage=stage, cond="ODDhits"
                )
                with open(FREQepochs_filename, "wb") as fp:
                    pickle.dump(FREQepochs, fp)
                with open(RAREepochs_filename, "wb") as fp:
                    pickle.dump(RAREepochs, fp)
                with open(ODDepochs_filename, "wb") as fp:
                    pickle.dump([FREQepochs, RAREepochs], fp)

            elif by == "resp":
                RESPepochs, NORESPepochs = split_trials(
                    BIDS_PATH, LOGS_DIR, subj=subj, run=run, stage=stage, by="resp"
                )
                RESPepochs_path, RESPepochs_filename = get_SAflow_bids(
                    BIDS_PATH, subj=subj, run=run, stage=stage, cond="RESP"
                )
                NORESPepochs_path, NORESPepochs_filename = get_SAflow_bids(
                    BIDS_PATH, subj=subj, run=run, stage=stage, cond="NORESP"
                )

                with open(RESPepochs_filename, "wb") as fp:
                    pickle.dump(RESPepochs, fp)
                with open(NORESPepochs_filename, "wb") as fp:
                    pickle.dump(NORESPepochs, fp)
