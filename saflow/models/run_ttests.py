from saflow import (
    BIDS_PATH,
    SUBJ_LIST,
    BLOCS_LIST,
    FREQS_NAMES,
    ZONE_CONDS,
    RESULTS_PATH,
    IMG_DIR,
)
import pickle
import saflow
from saflow.utils import array_topoplot, create_pval_mask, get_SAflow_bids
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    ShuffleSplit,
    LeaveOneGroupOut,
    KFold,
)
from sklearn.preprocessing import StandardScaler
import argparse
import os
import mne
import random
from tqdm import tqdm
import os.path as op
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm.autonotebook import tqdm
from saflow.models.run_classifs import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "-stage",
    "--stage",
    default="PSD",
    type=str,
    help="PSD files to use (PSD or PSD4001200)",
)
parser.add_argument(
    "-r",
    "--run",
    default="all",
    type=str,
    help="0 for all runs, all for all combinations",
)
parser.add_argument(
    "-f",
    "--frequency_band",
    default=None,
    type=str,
    help="Channels to compute",
)
parser.add_argument(
    "-p",
    "--n_permutations",
    default=1000,
    type=int,
    help="Number of permutations",
)
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
    help="Choose the classification problem ('VTC' or 'odd')",
)
parser.add_argument(
    "-l",
    "--level",
    default="group",
    type=str,
    help="Choose the classification level ('group' or 'subject')",
)
parser.add_argument(
    "-a",
    "--alpha",
    default=0.05,
    type=float,
    help="Desired alpha threshold",
)
parser.add_argument(
    "-cor",
    "--correction",
    default="fdr",
    type=str,
    help="Choose correction to apply",
)
parser.add_argument(
    "-avg",
    "--average",
    default=0,
    type=int,
    help="0 for no, 1 for yes",
)
parser.add_argument(
    "-norm",
    "--normalize",
    default=0,
    type=int,
    help="0 for no, 1 for yes",
)

args = parser.parse_args()


if __name__ == "__main__":
    split = args.split
    n_perms = args.n_permutations
    alpha = args.alpha
    by = args.by
    stage = args.stage
    normalize = args.normalize
    level = args.level
    if args.average == 0:
        avg = False
    elif args.average == 1:
        avg = True
    if not args.correction:
        correction = None
    else:
        correction = args.correction
    run = args.run
    if run == "0":
        run_name = "allruns"
        BLOCS_LIST = [saflow.BLOCS_LIST]  # if run is 0, compute everything
    elif run == "all":
        run_name = ["allruns", "2", "3", "4", "5", "6", "7"]
        BLOCS_LIST = [saflow.BLOCS_LIST, "2", "3", "4", "5", "6", "7"]
    else:
        BLOCS_LIST = [run]
        run_name = [run]
    if level == "group":
        SUBJ_LIST = [SUBJ_LIST]
        print("Processing all subjects.")
    elif level == "subject":
        SUBJ_LIST = SUBJ_LIST
        print(f"Processing subj-{SUBJ_LIST}")

    for run_idx, RUNS in enumerate(BLOCS_LIST):
        for SUBJ in SUBJ_LIST:
            if level == "group":
                if by == "VTC":
                    conds_list = (
                        ZONE_CONDS[0] + str(split[0]),
                        ZONE_CONDS[1] + str(split[1]),
                    )
                    savepath = op.join(
                        RESULTS_PATH,
                        f"VTC_ttest_{stage}_{n_perms}perm_{split[0]}{split[1]}_{correction}_avg{avg}_norm{normalize}_run-{run_name[run_idx]}/",
                    )

                elif by == "odd":
                    conds_list = ["FREQhits", "RAREhits"]
                    savepath = op.join(
                        RESULTS_PATH,
                        f"odd_ttest_{stage}_{n_perms}perm_{correction}_avg{avg}_norm{normalize}_run-{run_name[run_idx]}/",
                    )
                SUBJECTS = SUBJ
            elif level == "subject":
                if by == "VTC":
                    conds_list = (
                        ZONE_CONDS[0] + str(split[0]),
                        ZONE_CONDS[1] + str(split[1]),
                    )
                    savepath = op.join(
                        RESULTS_PATH,
                        f"VTC_ttest_{stage}_{n_perms}perm_{split[0]}{split[1]}_{correction}_avg{avg}_norm{normalize}_subj-{SUBJ}_run-{run_name[run_idx]}/",
                    )

                elif by == "odd":
                    conds_list = ["FREQhits", "RAREhits"]
                    savepath = op.join(
                        RESULTS_PATH,
                        f"odd_ttest_{stage}_{n_perms}perm_{correction}_avg{avg}_norm{normalize}_subj-{SUBJ}_run-{run_name[run_idx]}/",
                    )
                SUBJECTS = [SUBJ]
            print(savepath)
            if not (os.path.isdir(savepath)):
                os.makedirs(savepath)

            print(SUBJ)
            alltvals = []
            allcontrasts = []
            masks = []
            allpvals = []
            for FREQ in range(len(FREQS_NAMES)):

                savename = "PSD_ttest_{}.pkl".format(FREQS_NAMES[FREQ])

                condA_allchans = []
                condB_allchans = []
                X, y, groups = prepare_data(
                    BIDS_PATH,
                    SUBJECTS,
                    RUNS,
                    conds_list,
                    stage=stage,
                    CHAN=[x for x in range(270)],
                    FREQ=FREQ,
                    balance=True,
                    avg=avg,
                    level=level,
                )
                print(X[0, 0])
                if normalize:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                print(X[0, 0])
                condA = [x for i, x in enumerate(X) if y[i] == 0]
                condB = [x for i, x in enumerate(X) if y[i] == 1]
                condA_allchans = np.asarray(condA)
                condB_allchans = np.asarray(condB)
                print(f"cond {conds_list[0]} shape : {condA_allchans.shape}")
                print(f"cond {conds_list[1]} shape : {condB_allchans.shape}")

                if not avg:
                    tvals, pvals = stats.ttest_ind(
                        condA_allchans,
                        condB_allchans,  # cond1 = IN, cond2 = OUT
                        permutations=n_perms + 1,
                    )
                else:
                    tvals, pvals = stats.ttest_rel(condA_allchans, condB_allchans)
                pvals = fdrcorrection(pvals, alpha=alpha)[1]
                contrast = (condA_allchans - condB_allchans) / condB_allchans
                contrast = np.mean(contrast, axis=0)
                results = {"tvals": tvals, "pvals": pvals, "contrast": contrast}

                with open(op.join(savepath, savename), "wb") as f:
                    pickle.dump(results, f)
                print("Ok")
                print(f"Min pval : {min(pvals)}")

                allcontrasts.append(contrast)
                alltvals.append(results["tvals"])
                allpvals.append(results["pvals"])
                masks.append(create_pval_mask(results["pvals"], alpha=alpha))
