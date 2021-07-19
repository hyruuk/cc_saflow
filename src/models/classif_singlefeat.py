from src.saflow_params import BIDS_PATH, SUBJ_LIST, BLOCS_LIST, FREQS_NAMES, ZONE_CONDS, RESULTS_PATH
import pickle
from src.utils import get_SAflow_bids
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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
parser.add_argument(
    "-s",
    "--split",
    default=[25, 75],
    type=int,
    nargs='+',
    help="Bounds of percentile split",
)

#The arguments for the model selection can be :
#KNN for K neearest neighbors
#SVM for support vector machine
#DT for decision tree
#LR for Logistic Regression
parser.add_argument(
    "-m",
    "--model",
    default="LDA",
    type=str,
    help="Classifier to apply",
)

args = parser.parse_args()
model = args.model

def classif_singlefeat(X,y,groups, n_perms, model):

    if model == "LDA" :
        clf = LinearDiscriminantAnalysis()
    elif model == "KNN" :
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model == "SVM" :
        clf = SVC()
    elif model == "DT" : #For decision tree
        clf = DecisionTreeClassifier()
    elif model == "LR":
        clf = LogisticRegression()

    cv = LeaveOneGroupOut()
    results = classification(clf, cv, X, y, groups=groups, perm=n_perms, n_jobs=8)
    print('Done')
    print('DA : ' + str(results['acc_score']))
    print('p value : ' + str(results['acc_pvalue']))
    return results

def prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, CHAN=0, FREQ=0):
    # Prepare data
    X = []
    y = []
    groups = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for run in BLOCS_LIST:
            for i_cond, cond in enumerate(conds_list):
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
    split = args.split
    n_perms = args.n_permutations
    conds_list = (ZONE_CONDS[0] + str(split[0]), ZONE_CONDS[1] + str(split[1]))

    savepath = RESULTS_PATH + model + 'sf_LOGO_{}perm_{}{}/'.format(n_perms, split[0], split[1])

    if not(os.path.isdir(savepath)):
        os.makedirs(savepath)

    if args.channel != None:
        CHAN = args.channel
    if args.frequency_band != None:
        FREQ = FREQS_NAMES.index(args.frequency_band)
    if args.channel != None or args.frequency_band != None:
        savename = 'chan_{}_{}.pkl'.format(CHAN, FREQS_NAMES[FREQ])
        X, y, groups = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, CHAN=CHAN, FREQ=FREQ)
        result = classif_singlefeat(X,y, groups, n_perms=n_perms)
        with open(savepath + savename, 'wb') as f:
            pickle.dump(result, f)
    else:
        for CHAN in range(270):
            for FREQ in range(len(FREQS_NAMES)):
                savename = 'chan_{}_{}.pkl'.format(CHAN, FREQS_NAMES[FREQ])
                print(savename)
                if not(os.path.isfile(savepath + savename)):
                    X, y, groups = prepare_data(BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, CHAN=CHAN, FREQ=FREQ)
                    result = classif_singlefeat(X,y, groups, n_perms=n_perms, model=model)
                    with open(savepath + savename, 'wb') as f:
                        pickle.dump(result, f)
                print('Ok.')
