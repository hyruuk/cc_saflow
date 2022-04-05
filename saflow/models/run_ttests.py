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
from saflow.utils import array_topoplot, create_pval_mask, get_SAflow_bids
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    ShuffleSplit,
    LeaveOneGroupOut,
    KFold,
)
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from mlneurotools.stats import ttest_perm
import argparse
import os
import mne
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
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
    default=[25, 75],
    type=int,
    nargs="+",
    help="Bounds of percentile split",
)
parser.add_argument(
    "-by",
    "--by",
    default="VTC",
    type=str,
    help="Choose the classification problem ('VTC' or 'odd')",
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
    default=None,
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


args = parser.parse_args()


def prepare_data(
    BIDS_PATH,
    SUBJ_LIST,
    BLOCS_LIST,
    conds_list,
    CHAN=0,
    FREQ=0,
    balance=False,
    avg=True,
):
    # Prepare data
    X = []
    y = []
    groups = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for i_cond, cond in enumerate(conds_list):
            X_subj = []
            for run in BLOCS_LIST:
                _, fpath_cond = get_SAflow_bids(
                    BIDS_PATH, subj, run, stage="PSD", cond=cond
                )
                # print(f"subj-{subj}_run-0{run}_cond-{cond} chan : {CHAN} freq :{FREQ}")
                with open(fpath_cond, "rb") as f:
                    data = pickle.load(f)
                if avg:
                    X_subj.append(np.mean(data[:, CHAN, FREQ], axis=0))
                else:
                    for x in data[:, CHAN, FREQ]:
                        X.append(x)
                        y.append(i_cond)
                        groups.append(i_subj)
            if avg:
                X.append(np.mean(np.array(X_subj), axis=0))
                y.append(i_cond)
                groups.append(i_subj)
    if balance:
        X_balanced = []
        y_balanced = []
        groups_balanced = []
        # We want to balance the trials across subjects
        random.seed(10)
        for subj_idx in np.unique(groups):
            y_subj = [label for i, label in enumerate(y) if groups[i] == subj_idx]
            max_trials = min(np.unique(y_subj, return_counts=True)[1])

            X_subj_0 = [
                x for i, x in enumerate(X) if groups[i] == subj_idx and y[i] == 0
            ]
            X_subj_1 = [
                x for i, x in enumerate(X) if groups[i] == subj_idx and y[i] == 1
            ]

            idx_list_0 = [x for x in range(len(X_subj_0))]
            idx_list_1 = [x for x in range(len(X_subj_1))]
            picks_0 = random.sample(idx_list_0, max_trials)
            picks_1 = random.sample(idx_list_1, max_trials)

            for i in range(max_trials):
                X_balanced.append(X_subj_0[picks_0[i]])
                y_balanced.append(0)
                groups_balanced.append(subj_idx)
                X_balanced.append(X_subj_1[picks_1[i]])
                y_balanced.append(1)
                groups_balanced.append(subj_idx)

        X = X_balanced
        y = y_balanced
        groups = groups_balanced

    X = np.array(X)
    return X, y, groups


if __name__ == "__main__":
    split = args.split
    n_perms = args.n_permutations
    alpha = args.alpha
    by = args.by
    if args.average == 0:
        avg = False
    if args.average == 1:
        avg = True
    if not args.correction:
        correction = None
    else:
        correction = args.correction

    if by == "VTC":
        conds_list = (ZONE_CONDS[0] + str(split[0]), ZONE_CONDS[1] + str(split[1]))
        balance = False
        savepath = RESULTS_PATH + "VTC_ttest_{}perm_{}{}_{}_{}/".format(
            n_perms, split[0], split[1], correction, avg
        )
        figpath= op.join(IMG_DIR, f"{by}_tvals_{n_perms}perms_alpha{str(alpha)[2:]}_{split[0]}{split[1]}_{correction}_avg{avg}.png")
        figpath= op.join(IMG_DIR, f"{by}_contrast_{n_perms}perms_alpha{str(alpha)[2:]}_{split[0]}{split[1]}_{correction}_avg{avg}.png")
    elif by == "odd":
        conds_list = ["FREQhits", "RAREhits"]
        balance = True
        savepath = RESULTS_PATH + "{}_PSD_ttest_{}perm_{}__{}/".format(
            by, n_perms, correction, avg
        )
        figpath = op.join(IMG_DIR, f"{by}_tvals_{n_perms}perms_alpha{str(alpha)[2:]}_{correction}_avg{avg}.png")
        figpath_contrast = op.join(IMG_DIR, f"{by}_contrast_{n_perms}perms_alpha{str(alpha)[2:]}_{correction}_avg{avg}.png")

    if not (os.path.isdir(savepath)):
        os.makedirs(savepath)

    alltvals = []
    allcontrasts = []
    masks = []
    for FREQ in range(len(FREQS_NAMES)):
        savename = "PSD_ttest_{}.pkl".format(FREQS_NAMES[FREQ])
        condA_allchans = []
        condB_allchans = []
        # for CHAN in tqdm(range(270)):
        X, y, groups = prepare_data(
            BIDS_PATH,
            SUBJ_LIST,
            BLOCS_LIST,
            conds_list,
            FREQ=FREQ,
            CHAN=[x for x in range(270)],
            balance=True,
            avg=avg,
        )
        condA = [x for i, x in enumerate(X) if y[i] == 0]
        condB = [x for i, x in enumerate(X) if y[i] == 1]
        condA_allchans = np.asarray(condA)
        condB_allchans = np.asarray(condB)
        print(f"cond {conds_list[0]} shape : {condA_allchans.shape}")
        print(f"cond {conds_list[1]} shape : {condB_allchans.shape}")
        tvals, pvals = ttest_perm(
            condA_allchans,
            condB_allchans,  # cond1 = IN, cond2 = OUT
            n_perm=n_perms + 1,
            n_jobs=8,
            correction=correction,
            paired=False,
            two_tailed=True,
        )
        contrast = (condA_allchans - condB_allchans) / condB_allchans
        contrast = np.mean(contrast, axis=0)
        results = {"tvals": tvals, "pvals": pvals, "contrast": contrast}

        with open(savepath + savename, "wb") as f:
            pickle.dump(results, f)
        print("Ok")
        print(f"Min pval : {min(pvals)}")

        allcontrasts.append(contrast)
        alltvals.append(results["tvals"])
        masks.append(create_pval_mask(results["pvals"], alpha=alpha))

    # Plots
    # obtain chan locations
    _, data_fname = get_SAflow_bids(BIDS_PATH, subj="04", run="2", stage="-epo")
    epochs = mne.read_epochs(data_fname)
    ch_xy = epochs.pick_types(
        meg=True, ref_meg=False
    ).info  # Find the channel positions

    # Plot tvals
    toplot = alltvals
    vmax = np.max(np.max(abs(np.asarray(toplot))))
    vmin = -vmax
    array_topoplot(
        toplot,
        ch_xy,
        showtitle=True,
        titles=FREQS_NAMES,
        savefig=True,
        figpath=figpath,
        vmin=vmin,
        vmax=vmax,
        with_mask=True,
        masks=masks,
        cmap="magma",
    )

    # Plot contrasts
    toplot = allcontrasts
    vmax = np.max(np.max(abs(np.asarray(toplot))))
    vmin = -vmax
    array_topoplot(
        toplot,
        ch_xy,
        showtitle=True,
        titles=FREQS_NAMES,
        savefig=True,
        figpath=figpath_contrast,
        vmin=vmin,
        vmax=vmax,
        with_mask=True,
        masks=masks,
        cmap="coolwarm",
    )
