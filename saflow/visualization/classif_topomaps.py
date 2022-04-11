from saflow.utils import array_topoplot, create_pval_mask, get_SAflow_bids
from saflow import RESULTS_PATH, FREQS_NAMES, BIDS_PATH, IMG_DIR
from mlneurotools.stats import compute_pval
from str2bool import str2bool
import argparse
import mne
import itertools
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--classif_name",
    default="VTC_LDAsf_LOGO_1perm_2575",
    type=str,
    help="Name of the folder that contains classification results",
)
parser.add_argument(
    "-a",
    "--alpha",
    default=0.05,
    type=float,
    help="Desired alpha threshold",
)
args = parser.parse_args()

if __name__ == "__main__":
    classif_name = args.classif_name
    alpha = args.alpha
    savepath = RESULTS_PATH + classif_name + "/"

    # Load the data
    allfreqs_acc = []
    allfreqs_pval = []
    allmasks = []

    if "singlefeat" in classif_name:
        for FREQ in FREQS_NAMES:
            allchans_acc = []
            allchans_pval = []
            allchans_accperms = []
            for CHAN in range(270):
                savename = "freq_{}_chan_{}.pkl".format(FREQ, CHAN)
                with open(savepath + savename, "rb") as f:
                    result = pickle.load(f)
                allchans_acc.append(result["acc_score"])
                allchans_pval.append(result["acc_pvalue"])
                allchans_accperms.append(result["acc_pscores"])

            # Correction for multiple comparisons
            freq_perms = list(itertools.chain.from_iterable(allchans_accperms))
            corrected_pval = []
            for acc in allchans_acc:
                corrected_pval.append(compute_pval(acc, freq_perms))
            pval_mask = create_pval_mask(np.array(corrected_pval), alpha=alpha)

            allfreqs_acc.append(np.array(allchans_acc).squeeze())
            allfreqs_pval.append(np.array(allchans_pval).squeeze())
            allmasks.append(pval_mask)

        toplot_pval = allfreqs_pval
        figpath_pval = IMG_DIR + classif_name + "_pval" + str(alpha)[2:] + ".png"

        toplot_acc = allfreqs_acc
        figpath_acc = IMG_DIR + classif_name + "_acc" + str(alpha)[2:] + ".png"

        titles = FREQS_NAMES

    elif "_RF_" in classif_name:
        features_importances_list = []
        pvals_list = []
        permscores_list = []
        for FREQ in FREQS_NAMES:
            savename = "freq_{}.pkl".format(FREQS_NAMES[FREQ])
            with open(savepath + savename, "rb") as f:
                results = pickle.load(f)
        features_importances_list.append(results["feature_importances"])
        pvals_list.append(results["acc_pvalues"])
        permscores_list.append(results["acc_pscores"])
        # Correction for multiple comparisons
        # freq_perms = list(itertools.chain.from_iterable(allchans_accperms))
        toplot_featimp = features_importances_list
        figpath_featimp = (
            IMG_DIR + classif_name + "_features_importances" + str(alpha)[2:] + ".png"
        )

        freq_titles = []
        for idx_pval, pval in enumerate(pvals_list):
            if pval < alpha:
                freq_titles.append(FREQS_NAMES[idx_pval] + " *")
            else:
                freq_titles.append(FREQS_NAMES[idx_pval])

        titles = freq_titles

    _, data_fname = get_SAflow_bids(BIDS_PATH, subj="04", run="2", stage="-epo")
    epochs = mne.read_epochs(data_fname)
    ch_xy = epochs.pick_types(
        meg=True, ref_meg=False
    ).info  # Find the channel's position

    if "singlefeat" in classif_name:
        # Setup the min and max value of the color scale
        vmax_pval = np.max(np.max(np.asarray(toplot_pval)))
        vmin_pval = np.min(np.min(np.asarray(toplot_pval)))
        vmax_acc = np.max(np.max(np.asarray(toplot_acc)))
        vmin_acc = np.min(np.min(np.asarray(toplot_acc)))

        array_topoplot(
            toplot_pval,
            ch_xy,
            showtitle=True,
            titles=titles,
            savefig=True,
            figpath=figpath_pval,
            vmin=0,
            vmax=0.1,
            with_mask=True,
            masks=allmasks,
            cmap="viridis",
        )

        array_topoplot(
            toplot_acc,
            ch_xy,
            showtitle=True,
            titles=titles,
            savefig=True,
            figpath=figpath_acc,
            vmin=vmin_acc,
            vmax=vmax_acc,
            with_mask=True,
            masks=allmasks,
            cmap="plasma",
        )
    else:
        vmax_featimp = np.max(np.max(np.asarray(toplot_featimp)))
        array_topoplot(
            toplot_featimp,
            ch_xy,
            showtitle=True,
            titles=titles,
            savefig=True,
            figpath=figpath_featimp,
            vmin=0,
            vmax=vmax_featimp,
            with_mask=True,
            masks=None,
            cmap="red",
        )
