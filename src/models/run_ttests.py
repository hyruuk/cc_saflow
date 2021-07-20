from src.saflow_params import BIDS_PATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, ZONE_CONDS, RESULTS_PATH, IMG_DIR
import pickle
from src.utils import array_topoplot, create_pval_mask, get_SAflow_bids
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut, KFold
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from mlneurotools.stats import ttest_perm
import argparse
import os
import mne

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
    nargs='+',
    help="Bounds of percentile split",
)
parser.add_argument(
    "-a",
    "--alpha",
    default=0.05,
    type=float,
    help="Desired alpha threshold",
)

args = parser.parse_args()


def prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, FREQ=0):
    # Prepare data
    condA = []
    condB = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for run in BLOCS_LIST:
            for i_cond, cond in enumerate(conds_list):
                _, fpath_condA = get_SAflow_bids(BIDS_PATH, subj, run, stage='PSD', cond=cond)
                with open(fpath_condA, 'rb') as f:
                    data = pickle.load(f)
                if i_cond == 0:
                    for x in data[:, :, FREQ]:
                        condA.append(x)
                else:
                    for x in data[:, :, FREQ]:
                        condB.append(x)
    condA = np.array(condA)
    condB = np.array(condB)
    return condA, condB

if __name__ == "__main__":
    split = args.split
    n_perms = args.n_permutations
    alpha = args.alpha
    conds_list = (ZONE_CONDS[0] + str(split[0]), ZONE_CONDS[1] + str(split[1]))

    savepath = RESULTS_PATH + 'PSD_ttest_{}perm_{}{}/'.format(n_perms, split[0], split[1])
    figpath = IMG_DIR + 'PSD_ttest_{}perm_alpha{}_{}{}.png'.format(n_perms, str(alpha)[2:], split[0], split[1])
    if not(os.path.isdir(savepath)):
        os.makedirs(savepath)

    if args.frequency_band != None:
        FREQ = FREQS_NAMES.index(args.frequency_band)
    if args.frequency_band != None:
        savename = 'PSD_ttest_{}.pkl'.format(FREQS_NAMES[FREQ])
        condA, condB = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, FREQ=FREQ)
        tvals, pvals = ttest_perm(condA, condB, # cond1 = IN, cond2 = OUT
                n_perm=nperms+1,
                n_jobs=8,
                correction='maxstat',
                paired=False,
                two_tailed=True)
        print('Ok')
        results = {'tvals':tvals,
                   'pvals':pvals}
        with open(savepath + savename, 'wb') as f:
            pickle.dump(results, f)
    else:
        alltvals = []
        masks = []
        for FREQ in range(len(FREQS_NAMES)):
            savename = 'PSD_ttest_{}.pkl'.format(FREQS_NAMES[FREQ])
            if not(os.path.isfile(savepath + savename)):
                condA, condB = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, FREQ=FREQ)
                tvals, pvals = ttest_perm(condA, condB, # cond1 = IN, cond2 = OUT
                        n_perm=n_perms+1,
                        n_jobs=8,
                        correction='maxstat',
                        paired=False,
                        two_tailed=True)
                results = {'tvals':tvals,
                           'pvals':pvals}

                with open(savepath + savename, 'wb') as f:
                    pickle.dump(results, f)
                print('Ok')
            else:
                with open(savepath + savename, 'rb') as f:
                    results = pickle.load(f)
                    tvals = results['tvals']
            alltvals.append(results['tvals'])
            masks.append(create_pval_mask(results['pvals'], alpha=alpha))
        # plot
        toplot = alltvals
        vmax = np.max(np.max(abs(np.asarray(toplot))))
        vmin = -vmax
        # obtain chan locations
        _, data_fname = get_SAflow_bids(BIDS_PATH, subj='04', run='2', stage='-epo')
        epochs = mne.read_epochs(data_fname)
        ch_xy = epochs.pick_types(meg=True, ref_meg=False).info # Find the channel's position

        array_topoplot(toplot, ch_xy, showtitle=True, titles=FREQS_NAMES,
                        savefig=True, figpath=figpath, vmin=vmin, vmax=vmax,
                        with_mask=True, masks=masks, cmap='coolwarm')
