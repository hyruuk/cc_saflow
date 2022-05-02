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
from saflow.behav import get_VTC_from_file, plot_VTC, find_logfile
import os
from saflow.neuro import annotate_events, get_present_events

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


def annotate_precursor_events(BIDS_PATH, subj, bloc):

    _, epopath = get_SAflow_bids(BIDS_PATH, subj, bloc, stage="-epo4001200", cond=None)
    epochs = mne.read_epochs(epopath)
    # find events
    events = epochs.events
    files_list = os.listdir(LOGS_DIR)
    logfile = LOGS_DIR + find_logfile(subj, bloc, files_list)
    (
        IN_idx,
        OUT_idx,
        VTC_raw,
        VTC_filtered,
        IN_mask,
        OUT_mask,
        performance_dict,
        df_response,
        RT_to_VTC,
    ) = get_VTC_from_file(subj, bloc, files_list, inout_bounds=[50, 50])
    events = annotate_events(logfile, events, inout_idx=[IN_idx, OUT_idx])
    event_id = get_present_events(events)

    events_precursor = []
    for idx, ev in enumerate(events):
        if ev[2] == 310:
            if events[idx - 1][2] == 2111 or events[idx - 1][2] == 2110:
                events_precursor.append(events[idx - 1])
    events_precursor = np.array(events_precursor)

    epochs.events = events_precursor
    epochs.event_id = event_id
    return epochs


def new_split_trials(subj, run, by="VTC", inout_bounds=None):
    condA = []
    condB = []
    if by == "odd":
        cond = "5050"
    elif by == "VTC" or by == "VTCprec":
        cond = f"{inout_bounds[0]}{inout_bounds[1]}"
    for idx_freq, freq_bounds in enumerate(FREQS):
        _, PSDpath = get_SAflow_bids(
            BIDS_PATH,
            subj,
            run,
            stage=f"-epoenv4001200_{FREQS_NAMES[idx_freq]}",
            cond=cond,
        )

        epochs = mne.read_epochs(PSDpath)
        if by == "VTC":
            condA_epochs = epochs["FreqIN"]
            condB_epochs = epochs["FreqOUT"]
        elif by == "VTCprec":
            epochs_prec = annotate_precursor_events(BIDS_PATH, subj, run)
            epochs.events = epochs_prec.events
            epochs.event_id = epochs_prec.event_id
            print(epochs.events)
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
                    subj=subj, run=run, by="VTC", inout_bounds=CONDS_LIST
                )
                precINepochs, precOUTepochs = new_split_trials(
                    subj=subj, run=run, by="VTCprec", inout_bounds=CONDS_LIST
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
                _, precINepochs_filename = get_SAflow_bids(
                    BIDS_PATH,
                    subj=subj,
                    run=run,
                    stage=stage,
                    cond="precIN{}".format(CONDS_LIST[0]),
                )
                _, precOUTepochs_filename = get_SAflow_bids(
                    BIDS_PATH,
                    subj=subj,
                    run=run,
                    stage=stage,
                    cond="precOUT{}".format(CONDS_LIST[1]),
                )

                with open(INepochs_filename, "wb") as fp:
                    pickle.dump(INepochs, fp)
                with open(OUTepochs_filename, "wb") as fp:
                    pickle.dump(OUTepochs, fp)
                with open(precINepochs_filename, "wb") as fp:
                    pickle.dump(precINepochs, fp)
                with open(precOUTepochs_filename, "wb") as fp:
                    pickle.dump(precOUTepochs, fp)
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
