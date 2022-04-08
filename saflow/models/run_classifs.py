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
from numpy.random import permutation
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    ShuffleSplit,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    LeaveOneOut,
    StratifiedKFold,
    permutation_test_score,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from xgboost import XGBClassifier
from scipy.stats import uniform, zscore
from itertools import permutations
import argparse
import os
import random
import warnings
import pdb
from tqdm import tqdm
import os.path as op


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-subj",
    "--subject",
    default=None,
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-stage",
    "--stage",
    default="PSD",
    type=str,
    help="PSD files to use (PSD or PSD4001200)",
)
parser.add_argument(
    "-c",
    "--channel",
    default=None,
    type=int,
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
    default="VTC",
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
    "-avg",
    "--average",
    default=1,
    type=int,
    help="0 for no, 1 for yes",
)
parser.add_argument(
    "-norm",
    "--normalize",
    default=1,
    type=int,
    help="0 for no, 1 for yes",
)
parser.add_argument(
    "-mf",
    "--multifeatures",
    default=0,
    type=int,
    help="0 for no, 1 for yes",
)

# The arguments for the model selection can be :
# KNN for K nearest neighbors
# SVM for support vector machine
# DT for decision tree
# LR for Logistic Regression
# XGBC for XGBoost Classifier
parser.add_argument(
    "-m",
    "--model",
    default="LDA",
    type=str,
    help="Classifier to apply",
)

args = parser.parse_args()


def init_classifier(model_type="LDA"):
    if model == "LDA":
        clf = LinearDiscriminantAnalysis()
        distributions = dict()
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
        distributions = dict()
    elif model == "RF":
        clf = RandomForestClassifier()
        distributions = {
            "n_estimators": [100, 120, 150],
            "criterion": ["entropy", "gini"],
            "max_depth": [None, 1, 3, 5, 7, 9],
            "max_features": range(1, 11),
            "min_samples_split": range(2, 10),
            "min_samples_leaf": [1, 3, 5],
        }
    return clf, distributions


def apply_best_params(best_params, model):
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
    elif model == "RF":
        criterion = best_params["criterion"]
        max_depth = best_params["max_depth"]
        max_features = best_params["max_features"]
        min_samples_leaf = best_params["min_samples_leaf"]
        min_samples_split = best_params["min_samples_split"]
        n_estimators = best_params["n_estimators"]
        clf = RandomForestClassifier(
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
        )
    return clf


def classif_SKFold(X, y, n_perms, model, avg=0):
    # Find best parameters
    clf, distributions = init_classifier(model_type=model)

    if model != "XGBC" and model != "LDA" and avg == 0:
        # Optimize HPs
        outer_cv = StratifiedKFold()
        inner_cv = StratifiedKFold()
        best_params_list = []
        acc_score_list = []
        for train_outer, test_outer in outer_cv.split(X, y):
            search = RandomizedSearchCV(
                clf, distributions, cv=inner_cv, random_state=0
            ).fit(X[train_outer], y[train_outer])
            best_params = search.best_params_
            print("Best params : " + str(best_params))
            clf = apply_best_params(best_params, model)
            clf.fit(X[train_outer], y[train_outer])
            acc_score_outer = clf.score(X[test_outer], y[test_outer])
            acc_score_list.append(acc_score_outer)
            best_params_list.append(best_params)
            print("clf done :", acc_score_outer)

        # obtain hp of best DA
        best_fold_id = acc_score_list.index(max(acc_score_list))
        best_fold_params = best_params_list[best_fold_id]

        clf = apply_best_params(best_fold_params, model)
        score, permutation_scores, pvalue = permutation_test_score(
            clf, X, y, cv=outer_cv, n_permutations=n_perms, n_jobs=-1
        )
        results = {
            "acc_score": score,
            "acc_pscores": permutation_scores,
            "acc_pvalue": pvalue,
        }
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("p value : " + str(results["acc_pvalue"]))

    else:
        cv = StratifiedKFold()
        score, permutation_scores, pvalue = permutation_test_score(
            clf, X, y, cv=cv, n_permutations=n_perms, n_jobs=-1
        )
        results = {
            "acc_score": score,
            "acc_pscores": permutation_scores,
            "acc_pvalue": pvalue,
        }
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("p value : " + str(results["acc_pvalue"]))
    return results


def classif_LOGO(X, y, groups, n_cvgroups, n_perms, model, avg=0):

    clf, distributions = init_classifier(model_type=model)

    if model != "XGBC" and model != "LDA":  # and avg == 0:
        outer_cv = LeavePGroupsOut(n_groups=1)  # n_cvgroups)
        inner_cv = LeavePGroupsOut(n_groups=4)
        best_params_list = []
        acc_score_list = []
        for train_outer, test_outer in outer_cv.split(X, y, groups):
            search = RandomizedSearchCV(
                clf,
                distributions,
                cv=inner_cv,
                random_state=0,
                verbose=1,
            ).fit(X[train_outer], y[train_outer], groups[train_outer])
            best_params = search.best_params_
            print("Best params : " + str(best_params))
            clf = apply_best_params(best_params, model)
            clf.fit(X[train_outer], y[train_outer])
            acc_score_outer = clf.score(X[test_outer], y[test_outer])
            acc_score_list.append(acc_score_outer)
            best_params_list.append(best_params)
            print("clf done :", acc_score_outer)
        # obtain hp of best DA
        best_fold_id = acc_score_list.index(max(acc_score_list))
        best_fold_params = best_params_list[best_fold_id]
        clf = apply_best_params(best_fold_params, model)

        score, permutation_scores, pvalue = permutation_test_score(
            clf, X, y, groups=groups, cv=outer_cv, n_permutations=n_perms, n_jobs=-1
        )
        results = {
            "acc_score": score,
            "acc_pscores": permutation_scores,
            "acc_pvalue": pvalue,
        }
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("p value : " + str(results["acc_pvalue"]))

    else:
        cv = LeaveOneGroupOut()
        score, permutation_scores, pvalue = permutation_test_score(
            clf, X, y, groups=groups, cv=cv, n_permutations=n_perms, n_jobs=-1
        )
        results = {
            "acc_score": score,
            "acc_pscores": permutation_scores,
            "acc_pvalue": pvalue,
        }
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("p value : " + str(results["acc_pvalue"]))
    return results


def compute_pval(score, perm_scores):
    n_perm = len(perm_scores)
    pvalue = (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)
    return pvalue


def prepare_data(
    BIDS_PATH,
    SUBJ_LIST,
    BLOCS_LIST,
    conds_list,
    stage="PSD",
    CHAN=0,
    FREQ=None,
    balance=False,
    normalize=True,
    avg=False,
):
    if not FREQ:
        FREQ = [x for x in range(len(FREQS_NAMES))]
        singlefeat = False
    else:
        singlefeat = True
    # Prepare data
    X = []
    y = []
    groups = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for i_cond, cond in enumerate(conds_list):
            X_subj = []
            for run in BLOCS_LIST:
                _, fpath_cond = get_SAflow_bids(
                    BIDS_PATH, subj, run, stage=stage, cond=cond
                )
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
    if singlefeat:  # Not avg but mf
        X = np.array(X).reshape(-1, 1)
    else:
        X = np.array(X).reshape(-1, len(X_balanced[0]))  # ??
    y = np.asarray(y)
    groups = np.asarray(groups)

    if normalize:
        group_ids = np.unique(groups)
        for group_id in group_ids:
            X[groups == group_id] = zscore(X[groups == group_id], axis=0)
    return X, y, groups


if __name__ == "__main__":
    balance = True
    model = args.model
    split = args.split
    n_perms = args.n_permutations
    by = args.by
    level = args.level
    stage = args.stage
    if args.average == 0:
        avg = False
        average_string = "single-trial"
        n_cvgroups = 1
    elif args.average == 1:
        avg = True
        average_string = "averaged"
        n_cvgroups = 4
    if args.normalize == 0:
        normalize = False
        norm_string = "non-normalized"
    elif args.normalize == 1:
        normalize = True
        norm_string = "normalized"
    if args.multifeatures == 0:
        multifeatures = False
        mfsf_string = "singlefeat"
    elif args.multifeatures == 1:
        multifeatures = True
        mfsf_string = "multifeat"
    if level == "group":
        SUBJ_LIST = SUBJ_LIST
        print("Processing all subjects.")
    elif level == "subject":
        SUBJ_LIST = [args.subject]
        print(f"Processing subj-{args.subject}")

    if by == "VTC":
        conds_list = ("IN" + str(split[0]), "OUT" + str(split[1]))
    elif by == "odd":
        conds_list = ["FREQhits", "RAREhits"]
    elif by == "resp":
        conds_list = ["RESP", "NORESP"]
    if level == "group":
        foldername = f"{by}_{stage}_{model}_{level}-level_{mfsf_string}_{average_string}_{norm_string}_{n_perms}perm_{split[0]}{split[1]}-split"
    elif level == "subject":
        subject = args.subject
        foldername = f"{by}_{stage}_{model}_{level}-level_{mfsf_string}_{average_string}_{norm_string}_{n_perms}perm_{split[0]}{split[1]}-split_sub-{subject}"
    savepath = op.join(RESULTS_PATH, foldername)
    os.makedirs(savepath, exist_ok=True)
    print(foldername)
    if args.channel is not None:
        CHAN = args.channel
        if multifeatures:
            savename = "chan_{}.pkl".format(CHAN)
            print(savename)
            if not (os.path.isfile(op.join(savepath, savename))):
                X, y, groups = prepare_data(
                    BIDS_PATH,
                    SUBJ_LIST,
                    BLOCS_LIST,
                    conds_list,
                    stage=stage,
                    CHAN=CHAN,
                    balance=balance,
                    avg=avg,
                    normalize=normalize,
                )
                if level == "group":
                    result = classif_LOGO(
                        X,
                        y,
                        groups,
                        n_cvgroups=n_cvgroups,
                        n_perms=n_perms,
                        model=model,
                        avg=avg,
                    )
                else:
                    result = classif_SKFold(X, y, n_perms=n_perms, model=model, avg=avg)
                with open(op.join(savepath, savename), "wb") as f:
                    pickle.dump(result, f)
                print("Ok.")
        else:
            for FREQ in range(len(FREQS_NAMES)):
                savename = "chan_{}_{}.pkl".format(CHAN, FREQS_NAMES[FREQ])
                print(savename)
                if not (os.path.isfile(op.join(savepath, savename))):
                    X, y, groups = prepare_data(
                        BIDS_PATH,
                        SUBJ_LIST,
                        BLOCS_LIST,
                        conds_list,
                        stage=stage,
                        CHAN=CHAN,
                        FREQ=FREQ,
                        balance=balance,
                        avg=avg,
                        normalize=normalize,
                    )
                    if level == "group":
                        result = classif_LOGO(
                            X,
                            y,
                            groups,
                            n_cvgroups=n_cvgroups,
                            n_perms=n_perms,
                            model=model,
                            avg=avg,
                        )
                    else:
                        result = classif_SKFold(
                            X, y, n_perms=n_perms, model=model, avg=avg
                        )
                    with open(op.join(savepath, savename), "wb") as f:
                        pickle.dump(result, f)
                    print("Ok.")
    else:
        for CHAN in range(270):
            if multifeatures:
                savename = "chan_{}.pkl".format(CHAN)
                print(savename)
                if not (os.path.isfile(op.join(savepath, savename))):
                    X, y, groups = prepare_data(
                        BIDS_PATH,
                        SUBJ_LIST,
                        BLOCS_LIST,
                        conds_list,
                        stage=stage,
                        CHAN=CHAN,
                        balance=balance,
                        avg=avg,
                        normalize=normalize,
                    )
                    if level == "group":
                        result = classif_LOGO(
                            X,
                            y,
                            groups,
                            n_cvgroups=n_cvgroups,
                            n_perms=n_perms,
                            model=model,
                            avg=avg,
                        )
                    else:
                        result = classif_SKFold(
                            X, y, n_perms=n_perms, model=model, avg=avg
                        )
                    with open(op.join(savepath, savename), "wb") as f:
                        pickle.dump(result, f)
                    print("Ok.")
            else:
                for FREQ in range(len(FREQS_NAMES)):
                    savename = "chan_{}_{}.pkl".format(CHAN, FREQS_NAMES[FREQ])
                    print(savename)
                    if not (os.path.isfile(op.join(savepath, savename))):
                        X, y, groups = prepare_data(
                            BIDS_PATH,
                            SUBJ_LIST,
                            BLOCS_LIST,
                            conds_list,
                            stage=stage,
                            CHAN=CHAN,
                            FREQ=FREQ,
                            balance=balance,
                            avg=avg,
                            normalize=normalize,
                        )
                        if level == "group":
                            result = classif_LOGO(
                                X,
                                y,
                                groups,
                                n_cvgroups=n_cvgroups,
                                n_perms=n_perms,
                                model=model,
                                avg=avg,
                            )
                        else:
                            result = classif_SKFold(
                                X, y, n_perms=n_perms, model=model, avg=avg
                            )
                        with open(op.join(savepath, savename), "wb") as f:
                            pickle.dump(result, f)
                        print("Ok.")
