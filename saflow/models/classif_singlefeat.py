from saflow import (
    BIDS_PATH,
    SUBJ_LIST,
    BLOCS_LIST,
    FREQS_NAMES,
    ZONE_CONDS,
    RESULTS_PATH,
)
import pickle
from saflow.utils import get_SAflow_bids
import numpy as np
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    ShuffleSplit,
    LeaveOneGroupOut,
    KFold,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
import argparse
import os
import random
from scipy.stats import uniform

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

# The arguments for the model selection can be :
# KNN for K neearest neighbors
# SVM for support vector machine
# DT for decision tree
# LR for Logistic Regression
parser.add_argument(
    "-m",
    "--model",
    default="LDA",
    type=str,
    help="Classifier to apply",
)

args = parser.parse_args()


def classif_singlefeat(X, y, groups, n_perms, model):

    if model == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif model == "KNN":
        clf = KNeighborsClassifier()
        distributions = dict(
            n_neighbors=np.arange(1, 16, 1),
            weights=["uniform", "distance"],
            metric=["minkowski", "euclidean", "manhattan"],
        )
    elif model == "SVM":
        clf = SVC()
        distributions = dict()
    elif model == "DT":
        clf = DecisionTreeClassifier()
        distributions = dict(criterion=["gini", "entropy"], splitter=["best", "random"])
    elif model == "LR":
        clf = LogisticRegression()
        distributions = dict(
            C=uniform(loc=0, scale=4),
            penalty=["l2", "l1", "elasticnet", "none"],
            solver=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            multi_class=["auto", "ovr", "multinomial"],
        )
    elif model == "XGBC":
        clf = XGBClassifier()

    # Loop for permutations

    # Find best parameters
    if model != "XGBC" and model != "LDA":
        outer_cv = LeaveOneGroupOut()
        inner_cv = LeaveOneGroupOut()

        best_params_list = []
        acc_score_list = []

        for train_outer, test_outer in outer_cv.split(X, y, groups):
            DA_perm_list = []
            # Need to add the "fixed" randomized search
            search = RandomizedSearchCV(
                clf, distributions, cv=inner_cv, random_state=0
            ).fit(X[train_outer], y[train_outer], groups[train_outer])
            best_params = search.best_params_
            print("Best params : " + str(best_params))

            # Apply best hyperparameters
            if model == "KNN":
                metric = best_params["metric"]
                n_neighbors = best_params["n_neighbors"]
                weights = best_params["weights"]
                clf = KNeighborsClassifier(
                    n_neighbors=n_neighbors, metric=metric, weights=weights
                )
            elif model == "SVM":
                clf = SVC(best_params)
            elif model == "DT":
                criterion = best_params["criterion"]
                splitter = best_params["splitter"]
                clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            elif model == "LR":
                C = best_params["C"]
                penalty = best_params["penalty"]
                solver = best_params["solver"]
                multi_class = best_params["multi_class"]
                clf = LogisticRegression(
                    C=C, penalty=penalty, solver=solver, multi_class=multi_class
                )

            clf.fit(X[train_outer], y[train_outer])
            # evaluate fit above
            acc_score_outer = clf.score(X[train_outer], y[train_outer])
            # store hp and DA
            acc_score_list.append(acc_score_outer)
            best_params_list.append(best_params)
            print("clf done :", acc_score_outer)

        # obtain hp of best DA
        best_fold_id = acc_score_list.index(max(acc_score_list))
        best_fold_params = best_params_list[best_fold_id]

        # call arthur's classification() with best hp
        if model == "KNN":
            metric = best_fold_params["metric"]
            n_neighbors = best_fold_params["n_neighbors"]
            weights = best_fold_params["weights"]
            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors, metric=metric, weights=weights
            )
        elif model == "DT":
            criterion = best_fold_params["criterion"]
            splitter = best_fold_params["splitter"]
            clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
        elif model == "LR":
            C = best_fold_params["C"]
            penalty = best_fold_params["penalty"]
            solver = best_fold_params["solver"]
            multi_class = best_fold_params["multi_class"]
            clf = LogisticRegression(
                C=C, penalty=penalty, solver=solver, multi_class=multi_class
            )

        results = classification(
            clf, outer_cv, X, y, groups=groups, perm=n_perms, n_jobs=8
        )

        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("p value : " + str(results["acc_pvalue"]))

    else:
        inner_cv = LeaveOneGroupOut()
        results = classification(
            clf, inner_cv, X, y, groups=groups, perm=n_perms, n_jobs=8
        )
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("p value : " + str(results["acc_pvalue"]))

    return results


def prepare_data(
    BIDS_PATH, SUBJ_LIST, BLOCS_LIST, conds_list, CHAN=0, FREQ=0, balance=False
):
    # Prepare data
    X = []
    y = []
    groups = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for run in BLOCS_LIST:
            for i_cond, cond in enumerate(conds_list):
                _, fpath_condA = get_SAflow_bids(
                    BIDS_PATH, subj, run, stage="PSD", cond=cond
                )
                with open(fpath_condA, "rb") as f:
                    data = pickle.load(f)
                for x in data[:, CHAN, FREQ]:
                    X.append(x)
                    y.append(i_cond)
                    groups.append(i_subj)
    if balance:
        X_balanced = []
        y_balanced = []
        groups_balanced = []
        # We want to balance the trials across subjects
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
        y = np.asarray(y_balanced)
        groups = np.asarray(groups_balanced)
    X = np.array(X).reshape(-1, 1)
    return X, y, groups


if __name__ == "__main__":
    model = args.model
    split = args.split
    n_perms = args.n_permutations
    by = args.by
    if by == "VTC":
        conds_list = (ZONE_CONDS[0] + str(split[0]), ZONE_CONDS[1] + str(split[1]))
        balance = True
    elif by == "odd":
        conds_list = ["FREQhits", "RAREhits"]
        balance = True
    elif by == "resp":
        conds_list = ["RESP", "NORESP"]
        balance = True

    savepath = (
        RESULTS_PATH
        + "{}_".format(by)
        + model
        + "sf_LOGO_{}perm_{}{}/".format(n_perms, split[0], split[1])
    )

    if not (os.path.isdir(savepath)):
        os.makedirs(savepath)

    if args.channel != None:
        CHAN = args.channel
    if args.frequency_band != None:
        FREQ = FREQS_NAMES.index(args.frequency_band)
    if args.channel != None or args.frequency_band != None:
        savename = "chan_{}_{}.pkl".format(CHAN, FREQS_NAMES[FREQ])
        X, y, groups = prepare_data(
            BIDS_PATH,
            SUBJ_LIST,
            BLOCS_LIST,
            conds_list,
            CHAN=CHAN,
            FREQ=FREQ,
            balance=balance,
        )
        result = classif_singlefeat(X, y, groups, n_perms=n_perms, model=model)
        with open(savepath + savename, "wb") as f:
            pickle.dump(result, f)
    else:
        for CHAN in range(270):
            for FREQ in range(len(FREQS_NAMES)):
                savename = "chan_{}_{}.pkl".format(CHAN, FREQS_NAMES[FREQ])
                print(savename)
                if not (os.path.isfile(savepath + savename)):
                    X, y, groups = prepare_data(
                        BIDS_PATH,
                        SUBJ_LIST,
                        BLOCS_LIST,
                        conds_list,
                        CHAN=CHAN,
                        FREQ=FREQ,
                        balance=balance,
                    )
                    result = classif_singlefeat(
                        X, y, groups, n_perms=n_perms, model=model
                    )
                    with open(savepath + savename, "wb") as f:
                        pickle.dump(result, f)
                print("Ok.")
