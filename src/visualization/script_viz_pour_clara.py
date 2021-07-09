from src.utils import array_topoplot, create_pval_mask, get_SAflow_bids
from src.saflow_params import RESULTS_PATH, FREQS_NAMES, BIDS_PATH, IMG_DIR
from mlneurotools.stats import compute_pval
from str2bool import str2bool #Need to be installed
import argparse
import mne
import itertools
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--classif_name",
    default='LDAsf_LOGO_100perm',
    type=str,
    help="Name of the folder that contains classification results",
)
parser.add_argument("--pval", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Plot p-value. If False, plot Decoding Accuracy instead")
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
    pval = args.pval
    alpha = args.alpha
    savepath = RESULTS_PATH + classif_name + '/'

    # Load the data
    allfreqs_acc = []
    allfreqs_pval = []
    allmasks = []
    for FREQ in FREQS_NAMES:
        allchans_acc = []
        allchans_pval = []
        allchans_accperms = []
        for CHAN in range(270):
            savename = 'chan_{}_{}.pkl'.format(CHAN, FREQ)
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
            n_perm = len(freq_perms)
            pvalue = (np.sum(freq_perms >= acc) + 1.0) / (n_perm + 1)
        pval_mask = create_pval_mask(np.array(corrected_pval), alpha=alpha)
        
        allfreqs_acc.append(np.array(allchans_acc).squeeze()) #array, shape (n_chan,)
        allfreqs_pval.append(np.array(allchans_pval).squeeze()) #array, shape (n_chan,)
        allmasks.append(pval_mask)

    if pval:
        toplot = allfreqs_pval
        figpath = IMG_DIR + classif_name + '_pval.png'
    else:
        toplot = allfreqs_acc
        figpath = IMG_DIR + classif_name + '_acc.png'

    _, data_fname = get_SAflow_bids(BIDS_PATH, subj='04', run='2', stage='-epo')
    epochs = mne.read_epochs(data_fname)
    ch_xy = epochs.pick_types(meg=True, ref_meg=False).info #type : np.ndarray, shape : (n_chan, 2)

    vmax = np.max(np.max(np.asarray(toplot)))
    vmin = np.min(np.min(np.asarray(toplot)))

    array_topoplot(toplot, ch_xy, showtitle=True, titles=FREQS_NAMES,
                    savefig=True, figpath=figpath, vmin=vmin, vmax=vmax,
                    with_mask=True, masks=allmasks)
