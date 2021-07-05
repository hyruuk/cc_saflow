tfrom src.saflow_params import BIDS_PATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, ZONE_CONDS, RESULTS_PATH
import pickle
from src.utils import get_SAflow_bids
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--channel",
    default=None,
    type=int,
    help="Channels to compute",
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

args = parser.parse_args()


def classif_singlefeat(X,y,groups, n_perms):
    cv = LeaveOneGroupOut()
    clf = LinearDiscriminantAnalysis()
    results = classification(clf, cv, X, y, groups=groups, perm=n_perms, n_jobs=8)
    print('Done')
    print('DA : ' + str(results['acc_score']))
    print('p value : ' + str(results['acc_pvalue']))
    return results

def prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, CONDS_LIST, CHAN=0, FREQ=0):
    # Prepare data
    X = []
    y = []
    groups = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for run in BLOCS_LIST:
            for i_cond, cond in enumerate(CONDS_LIST):
                _, fpath_condA = get_SAflow_bids(BIDS_PATH, subj, run, stage='PSD', cond=cond)
                with open(fpath_condA, 'rb') as f:
                    data = pickle.load(f)
                for x in data[:, CHAN, FREQ]:
                    X.append(x)
                    y.append(i_cond)
                    groups.append(i_subj)
    X = np.array(X).reshape(-1, 1)
    return X, y, groups

if __name__ == "__main__":
    CONDS_LIST = ZONE_CONDS
    N_PERMS = args.n_permutations

    savepath = RESULTS_PATH + '/LDAsf_LOGO_{}perm/'.format(N_PERMS)
    if not(os.path.isdir(savepath)):
        os.makedirs(savepath)

    if args.channel != None:
        CHAN = args.channel
    if args.frequency_band != None:
        FREQ = FREQS_NAMES.index(args.frequency_band)
    if args.channel != None or args.frequency_band != None:
        X, y, groups = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, CONDS_LIST, CHAN=CHAN, FREQ=FREQ)
        result = classif_singlefeat(X,y, groups, n_perms=N_PERMS)
        savename = 'chan_{}_{}.pkl'.format(CHAN, FREQS_NAMES[FREQ])
        with open(savepath + savename, 'wb') as f:
            pickle.dump(result, f)
    else:
        for CHAN in range(270):
            for FREQ in range(len(FREQS_NAMES)):
                savename = 'chan_{}_{}.pkl'.format(CHAN, FREQS_NAMES[FREQ])
                print(savename)
                if not(os.path.isfile(savepath + savename)):
                    X, y, groups = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, CONDS_LIST, CHAN=CHAN, FREQ=FREQ)
                    result = classif_singlefeat(X,y, groups, n_perms=N_PERMS)
                    with open(savepath + savename, 'wb') as f:
                        pickle.dump(result, f)
                print('Ok.')
