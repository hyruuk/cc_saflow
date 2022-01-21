from saflow.utils import array_topoplot, create_pval_mask, get_SAflow_bids
from saflow.saflow_params import RESULTS_PATH, BIDS_PATH, IMG_DIR
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
    default='LDAmf_LOGO_100perm',
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
parser.add_argument(
    "-s",
    "--subject",
    default=None,
    type=str,
    help="Subject to plot (if none, plot between-subject classifs)",
)
args = parser.parse_args()


if __name__ == "__main__":
    classif_name = args.classif_name
    alpha = args.alpha
    subject = args.subject
    savepath = RESULTS_PATH + classif_name + '/'

    # Load the data
    allfreqs_acc = []
    allfreqs_pval = []
    allmasks = []

    allchans_acc = []
    allchans_pval = []
    allchans_accperms = []
    for CHAN in range(270):

        with open(savepath + savename, 'rb') as f:
            result = pickle.load(f)
        allchans_acc.append(result['acc_score'])
        allchans_pval.append(result['acc_pvalue'])
        allchans_accperms.append(result['acc_pscores'])

    # Correction for multiple comparisons
    freq_perms = list(itertools.chain.from_iterable(allchans_accperms))
    corrected_pval = []
    for acc in allchans_acc:
        corrected_pval.append(compute_pval(acc[0], freq_perms))
    pval_mask = create_pval_mask(np.array(corrected_pval), alpha=alpha)

    allfreqs_acc.append(np.array(allchans_acc).squeeze()) #array, shape (n_chan,)
    allfreqs_pval.append(np.array(allchans_pval).squeeze()) #array, shape (n_chan,)
    allmasks.append(pval_mask)

    toplot_pval = allfreqs_pval
    figpath_pval = IMG_DIR + classif_name + 'mf_pval' + str(alpha)[2:] + '.png'

    toplot_acc = allfreqs_acc
    figpath_acc = IMG_DIR + classif_name + 'mf_acc' + str(alpha)[2:] + '.png'

    _, data_fname = get_SAflow_bids(BIDS_PATH, subj='04', run='2', stage='-epo')
    epochs = mne.read_epochs(data_fname)
    ch_xy = epochs.pick_types(meg=True, ref_meg=False).info # Find the channel's position

    # Setup the min and max value of the color scale
    vmax_pval = np.max(np.max(np.asarray(toplot_pval)))
    vmin_pval = np.min(np.min(np.asarray(toplot_pval)))
    vmax_acc = np.max(np.max(np.asarray(toplot_acc)))
    vmin_acc = np.min(np.min(np.asarray(toplot_acc)))

    TITLE =['All Frequencies'] #I tied other methods but it didn't work...
    array_topoplot(toplot_pval, ch_xy, showtitle=True, titles=TITLE,
                    savefig=True, figpath=figpath_pval, vmin=vmin_pval, vmax=vmax_pval,
                    with_mask=True, masks=allmasks)

    array_topoplot(toplot_acc, ch_xy, showtitle=True, titles=TITLE,
                    savefig=True, figpath=figpath_acc, vmin=vmin_acc, vmax=vmax_acc,
                    with_mask=True, masks=allmasks)
