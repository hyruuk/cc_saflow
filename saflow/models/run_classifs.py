from saflow import (
    BIDS_PATH,
    SUBJ_LIST,
    BLOCS_LIST,
    FREQS_NAMES,
    ZONE_CONDS,
    RESULTS_PATH,
)
import saflow
import pickle
from saflow.utils import get_SAflow_bids
import numpy as np
from numpy.random import permutation
from sklearn.preprocessing import StandardScaler
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
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from mlneurotools.ml import classification, StratifiedShuffleGroupSplit
from xgboost import XGBClassifier
from scipy.stats import uniform, zscore, loguniform
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
    "-f",
    "--freq",
    default=None,
    type=str,
    help="Freq band to process (ex. alpha)",
)
parser.add_argument(
    "-r",
    "--run",
    default="all",
    type=str,
    help="0 for all runs, all for all combinations",
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
    default="LR",
    type=str,
    help="Classifier to apply",
)

args = parser.parse_args()


def init_classifier(model="LDA"):
    if model == "LDA":
        clf = LinearDiscriminantAnalysis()
        distributions = dict()
    elif model == "KNN":
        clf = KNeighborsClassifier()
        distributions = dict(
            classifier__n_neighbors=np.arange(1, 16, 1),
            classifier__weights=["uniform", "distance"],
            classifier__metric=["minkowski", "euclidean", "manhattan"],
        )
    elif model == "SVM":
        clf = SVC(kernel="linear")
        distributions = {
            "classifier__C": [
                0.1,
                0.5,
                1,
                3,
                10,
                50,
                100,
                200,
                500,
                1000,
            ],  # uniform(loc=0, scale=100),
            "classifier__gamma": [5, 2, 1, 0.01, 0.001, 0.0001, 0.00001],
            "classifier__kernel": ["linear"],  # "rbf", "poly", "sigmoid", "linear"],
            "classifier__max_iter": [100, 500, 1000],  # , 200, 300, 400, 500, 1000],
        }
    elif model == "DT":
        clf = DecisionTreeClassifier()
        distributions = dict(
            classifier__criterion=["gini", "entropy"],
            classifier__splitter=["best", "random"],
        )
    elif model == "LR":
        clf = LogisticRegression()
        distributions = dict(
            classifier__C=uniform(loc=0, scale=4),
            classifier__penalty=["l2", "l1", "elasticnet", "none"],
            classifier__solver=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            classifier__multi_class=["auto", "ovr", "multinomial"],
            classifier__max_iter=[100, 200, 300, 400, 500, 1000],
        )
    elif model == "XGBC":
        clf = XGBClassifier()
        distributions = dict()
    elif model == "RF":
        clf = RandomForestClassifier()
        distributions = {
            "classifier__n_estimators": [10],  # mettre dautres valeurs
            "classifier__max_depth": [1, 2, 4, 8, 12, 16],
            "classifier__min_samples_split": [2, 4, 6, 8, 10, 12, 14, 16],
            "classifier__max_features": [0.25],
        }
    return clf, distributions


def apply_best_params(best_params, model):
    # Apply best hyperparameters
    if model == "KNN":
        metric = best_params["classifier__metric"]
        n_neighbors = best_params["classifier__n_neighbors"]
        weights = best_params["classifier__weights"]
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, weights=weights
        )
    elif model == "SVM":
        C = best_params["classifier__C"]
        gamma = best_params["classifier__gamma"]
        kernel = best_params["classifier__kernel"]
        max_iter = best_params["classifier__max_iter"]
        clf = SVC(C=C, gamma=gamma, kernel=kernel, max_iter=max_iter)
    elif model == "DT":
        criterion = best_params["classifier__criterion"]
        splitter = best_params["classifier__splitter"]
        clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
    elif model == "LR":
        C = best_params["classifier__C"]
        penalty = best_params["classifier__penalty"]
        solver = best_params["classifier__solver"]
        multi_class = best_params["classifier__multi_class"]
        max_iter = best_params["classifier__max_iter"]
        clf = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            multi_class=multi_class,
            max_iter=max_iter,
        )
    elif model == "RF":
        max_depth = best_params["classifier__max_depth"]
        max_features = best_params["classifier__max_features"]
        min_samples_split = best_params["classifier__min_samples_split"]
        n_estimators = best_params["classifier__n_estimators"]
        clf = RandomForestClassifier(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
        )
    return clf


def classif_LOGO(X, y, groups, n_cvgroups, n_perms, model, avg=0, norm=1):

    clf, distributions = init_classifier(model)
    if norm == 1:
        scaler = StandardScaler()
        pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])
    else:
        pipeline = clf

    if model != "XGBC" and model != "LDA" and avg == 0:
        if groups is None:
            outer_cv = StratifiedKFold()
            inner_cv = StratifiedKFold()
        else:
            outer_cv = LeavePGroupsOut(n_groups=1)  # n_cvgroups)
            inner_cv = LeavePGroupsOut(n_groups=1)
            # outer_cv = GroupShuffleSplit(n_splits=10, test_size=1)
            # inner_cv = GroupShuffleSplit(n_splits=10, test_size=2)

        best_params_list = []
        acc_score_list = []
        for train_outer, test_outer in outer_cv.split(X, y, groups):
            rs_obj = RandomizedSearchCV(
                pipeline,
                distributions,
                cv=inner_cv,
                random_state=0,
                verbose=1,
            )
            if groups is None:
                search = rs_obj.fit(X[train_outer], y[train_outer])
            else:
                search = rs_obj.fit(X[train_outer], y[train_outer], groups[train_outer])
            best_params = search.best_params_
            print("Best params : " + str(best_params))
            clf = apply_best_params(best_params, model)
            if norm == 1:
                scaler = StandardScaler()
                pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])
            else:
                pipeline = clf
            pipeline.fit(X[train_outer], y[train_outer])
            acc_score_outer = pipeline.score(X[test_outer], y[test_outer])
            acc_score_list.append(acc_score_outer)
            best_params_list.append(best_params)
            print("clf done :", acc_score_outer)
        # obtain hp of best DA
        best_fold_id = acc_score_list.index(max(acc_score_list))
        best_fold_params = best_params_list[best_fold_id]
        clf = apply_best_params(best_fold_params, model)
        if norm == 1:
            scaler = StandardScaler()
            pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])
        else:
            pipeline = clf

        results = final_classif(
            pipeline, outer_cv, X, y, groups, model, norm, n_perms=n_perms
        )
        results["best_params"] = best_fold_params
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("DA on train set : " + str(results["DA_train"]))
        print("p value : " + str(results["acc_pvalue"]))
    else:
        if groups is None:
            cv = StratifiedKFold()
        else:
            cv = LeaveOneGroupOut()
        results = final_classif(
            pipeline, cv, X, y, groups, model, norm, n_perms=n_perms
        )
        print("Done")
        print("DA : " + str(results["acc_score"]))
        print("DA on train set : " + str(results["DA_train"]))
        print("p value : " + str(results["acc_pvalue"]))
    return results


def final_classif(pipeline, cv, X, y, groups, model, norm, n_perms=1000):
    score, permutation_scores, pvalue = permutation_test_score(
        pipeline, X, y, groups=groups, cv=cv, n_permutations=n_perms, n_jobs=-1
    )
    results = {
        "acc_score": score,
        "acc_pscores": permutation_scores,
        "acc_pvalue": pvalue,
    }
    # Get DA train and feature importance
    pipeline.fit(X, y)
    results["DA_train"] = pipeline.score(X, y)
    if model == "RF":
        if norm == 1:
            results["feature_importances"] = pipeline["classifier"].feature_importances_
        else:
            results["feature_importances"] = pipeline.feature_importances_
    elif model == "LR" or model == "SVM":
        if norm == 1:
            results["feature_importances"] = pipeline["classifier"].coef_.squeeze()
        else:
            results["feature_importances"] = pipeline.coef_.squeeze()
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
    level="group",
):
    # Prepare data
    X = []
    y = []
    groups = []
    for i_subj, subj in enumerate(SUBJ_LIST):
        for i_cond, cond in enumerate(conds_list):
            X_subj = []
            for i_run, run in enumerate(BLOCS_LIST):
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
                        if level == "group":
                            groups.append(i_subj)
                        elif level == "subject":
                            groups.append(i_run)
            if avg:
                X.append(np.mean(np.array(X_subj), axis=0))
                y.append(i_cond)
                groups.append(i_subj)
    if balance:
        X, y, groups = balance_data(X, y, groups)

    if type(CHAN) is int and type(FREQ) is int:
        X = np.array(X).reshape(-1, 1)
    elif len(CHAN) != 1:  #
        X = np.array(X).reshape(-1, len(X[0]))  # ??
    y = np.asarray(y)
    groups = np.asarray(groups)
    return X, y, groups


def balance_data(X, y, groups, seed=10):
    X_balanced = []
    y_balanced = []
    groups_balanced = []
    # We want to balance the trials across subjects
    random.seed(seed)
    for subj_idx in np.unique(groups):
        y_subj = [label for i, label in enumerate(y) if groups[i] == subj_idx]
        max_trials = min(np.unique(y_subj, return_counts=True)[1])

        X_subj_0 = [x for i, x in enumerate(X) if groups[i] == subj_idx and y[i] == 0]
        X_subj_1 = [x for i, x in enumerate(X) if groups[i] == subj_idx and y[i] == 1]

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
    return X, y, groups


if __name__ == "__main__":
    model = args.model
    split = args.split
    n_perms = args.n_permutations
    by = args.by
    level = args.level
    stage = args.stage
    run = args.run
    if args.average == 0:
        avg = False
        average_string = "single-trial"
        n_cvgroups = 1
    elif args.average == 1:
        avg = True
        average_string = "averaged"
        n_cvgroups = 4
    if args.normalize == 0:
        normalize = 0
        norm_string = "non-normalized"
    elif args.normalize == 1:
        normalize = 1
        norm_string = "normalized"
    if level == "group":
        SUBJ_LIST = SUBJ_LIST
        print("Processing all subjects.")
    elif level == "subject":
        SUBJ_LIST = [args.subject]
        print(f"Processing subj-{args.subject}")
    if run == "0":
        run = "allruns"
        BLOCS_LIST = [saflow.BLOCS_LIST]  # if run is 0, compute everything
    elif run == "all":
        run = ["allruns", "2", "3", "4", "5", "6", "7"]
        BLOCS_LIST = [saflow.BLOCS_LIST, "2", "3", "4", "5", "6", "7"]
    else:
        BLOCS_LIST = [[run]]
    if args.channel is not None:
        CHANS = [args.channel]
    else:
        CHANS = [x for x in range(270)]
    if args.multifeatures == 0:
        multifeatures = False
        mfsf_string = "singlefeat"
    elif args.multifeatures == 1:
        multifeatures = True
        mfsf_string = "multifeat"
        assert args.channel is None, "Channels should be None for multifeatures"
        CHANS = [CHANS]
    if args.freq is not None:
        FREQS = [FREQS_NAMES.index(args.freq)]
    else:
        FREQS = [x for x in range(len(FREQS_NAMES))]

    # Generate string names
    if by == "VTC":
        conds_list = ("IN" + str(split[0]), "OUT" + str(split[1]))
    elif by == "odd":
        conds_list = ["FREQhits", "RAREhits"]
    elif by == "resp":
        conds_list = ["RESP", "NORESP"]
    for run_idx, BLOCS in enumerate(BLOCS_LIST):
        if level == "group":
            foldername = f"{by}_{stage}_{model}_{level}-level_{mfsf_string}_{average_string}_{norm_string}_{n_perms}perm_{split[0]}{split[1]}-split_run-{run[run_idx]}"
        elif level == "subject":
            subject = args.subject
            foldername = f"{by}_{stage}_{model}_{level}-level_{mfsf_string}_{average_string}_{norm_string}_{n_perms}perm_{split[0]}{split[1]}-split_sub-{subject}_run-{[run_idx]}"
        savepath = op.join(RESULTS_PATH, foldername)
        os.makedirs(savepath, exist_ok=True)
        print(foldername)
        for FREQ in FREQS:
            for CHAN in CHANS:
                if multifeatures:
                    savename = "freq_{}.pkl".format(FREQS_NAMES[FREQ])
                else:
                    savename = "freq_{}_chan_{}.pkl".format(FREQS_NAMES[FREQ], CHAN)
                print(savename)
                X, y, groups = prepare_data(
                    BIDS_PATH,
                    SUBJ_LIST,
                    BLOCS,
                    conds_list,
                    stage=stage,
                    CHAN=CHAN,
                    FREQ=FREQ,
                    balance=False,
                    avg=avg,
                    normalize=normalize,
                    level=level,
                )
                print(f"X shape : {X.shape}")
                print(f"y shape : {y.shape}")
                print(f"groups shape : {groups.shape}")
                results = classif_LOGO(
                    X,
                    y,
                    groups,
                    n_cvgroups=n_cvgroups,
                    n_perms=n_perms,
                    model=model,
                    avg=avg,
                    norm=normalize,
                )
                with open(op.join(savepath, savename), "wb") as f:
                    pickle.dump(results, f)
                print("Ok.")
