from src.saflow_params import BIDS_PATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, ZONE_CONDS, RESULTS_PATH
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
    default=0,
    type=int,
    help="Channels to compute",
)
parser.add_argument(
    "-f",
    "--frequency_band",
    default='alpha',
    type=str,
    help="Channels to compute",
)
args = parser.parse_args()


def classif_singlefeat(X,y,groups):
    cv = LeaveOneGroupOut()
    clf = LinearDiscriminantAnalysis()
    results = classification(clf, cv, X, y, groups=groups, perm=1000, n_jobs=8)
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

if __name__ == "__main__":
    CONDS_LIST = ZONE_CONDS
    #CHAN = args.channel
    #FREQ = FREQS_NAMES.index(args.frequency_band)
    savepath = RESULTS_PATH + '/LDAsf_LOGO_1000perm/'
    if not(os.path.isdir(savepath)):
        os.makedirs(savepath)

    for CHAN in range(270):
        for FREQ in range(len(FREQS_NAMES)):
            print(FREQ)
            #X, y, groups = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, CONDS_LIST, CHAN=CHAN, FREQ=FREQ)
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
            result = classif_singlefeat(X,y, groups)
            savename = 'chan_{}_{}.pkl'.format(CHAN, FREQS_NAMES[FREQ])
            with open(savepath + savename, 'wb') as f:
                pickle.dump(result, f)
