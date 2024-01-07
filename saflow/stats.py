import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from fooof import FOOOF, Bands, FOOOFGroup
import mne_bids
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def singlefeat_classif(X, y, groups, clf=LinearDiscriminantAnalysis(), cv=LeaveOneGroupOut(), n_perms=1):
    all_scores, all_perm_scores, all_pvals = [], [], []
    for freq_idx in range(X.shape[0]):
        scores, perm_scores, pvals = [], [], []
        for chan_idx in range(X.shape[-1]):
            X_sf = X[freq_idx,:,chan_idx]
            score, permutation_scores, pvalue = permutation_test_score(clf, 
                                                                       X=X_sf.reshape(-1, 1), 
                                                                       y=y, 
                                                                       groups=groups, 
                                                                       cv=cv, 
                                                                       n_permutations=n_perms, 
                                                                       scoring='roc_auc', 
                                                                       n_jobs=-1)
            scores.append(score)
            perm_scores.append(permutation_scores)
            pvals.append(pvalue)
            print(f'Computed feature {freq_idx} chan {chan_idx} Score {score}, pvalue {pvalue}')
        all_scores.append(scores)
        all_perm_scores.append(perm_scores)
        all_pvals.append(pvals)
        
    all_scores = np.array(all_scores)
    all_perm_scores = np.array(all_perm_scores)
    all_pvals = np.array(all_pvals)
    all_results = {'scores': all_scores, 'perm_scores': all_perm_scores, 'pvals': all_pvals}
    return all_results

def subject_average(X, y, groups):
    """Computes group-averages."""
    new_X = []
    new_y = []
    for subj in np.unique(groups):
        for cond in np.unique(y):
            new_X.append(np.nanmedian(X.transpose(1,0,2)[(groups == subj) & (y == cond)], axis=0))
            new_y.append(cond)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return new_X, new_y

def simple_contrast(X, y, groups):
    """Computes subject-averages and contrasts between conditions."""
    n_features = X.shape[0]
    X_avg, y_avg = subject_average(X, y, groups)
    # Average each condition separately
    X_avg_by_cond = []
    for cond in np.unique(y_avg):
        X_avg_by_cond.append(np.nanmean(X_avg[y_avg == cond], axis=0))
    # Compute normalized contrast (A - B)/B
    X_contrast = (X_avg_by_cond[0] - X_avg_by_cond[1]) / X_avg_by_cond[1]
    # Split conditions for ttest
    X_condA = X_avg[y_avg == 0]
    X_condB = X_avg[y_avg == 1]
    
    # Compute t-test
    tvals = []
    pvals = []
    for feature_idx in range(n_features):
        t, p = stats.ttest_ind(X_condB[:,feature_idx,:], X_condA[:,feature_idx,:], axis=0)
        tvals.append(t)
        pvals.append(p)
    tvals = np.array(tvals)
    pvals = np.array(pvals)
    return X_contrast, tvals, pvals

def subject_contrast(X, y):
    n_features = X.shape[0]
    X_avg_by_cond = []
    X = X.transpose(1,0,2)
    for cond in np.unique(y):
        X_avg_by_cond.append(np.nanmean(X[y == cond], axis=0))
    # Compute normalized contrast (A - B)/B
    X_contrast = (X_avg_by_cond[0] - X_avg_by_cond[1]) / X_avg_by_cond[1]
    # Split conditions for ttest
    X_condA = X[y == 0]
    X_condB = X[y == 1]
    tvals = []
    pvals = []
    for feature_idx in range(n_features):
        t, p = stats.ttest_rel(X_condB[:,feature_idx,:], X_condA[:,feature_idx,:], axis=0)
        tvals.append(t)
        pvals.append(p)
    tvals = np.array(tvals)
    pvals = np.array(pvals)
    return X_contrast, tvals, pvals

# Create significance mask
def mask_pvals(pvals, alpha):
    return pvals < alpha

def apply_tmax(all_results):
    da = all_results['scores']

    da_perms = all_results['perm_scores']
    da_perms = da_perms.reshape(da_perms.shape[0], -1)

    tmax_pvals = np.empty_like(da)
    for x in range(da.shape[0]):
        for y in range(da.shape[1]):
            pval = compute_pval(da[x,y], da_perms[x,:])
            tmax_pvals[x,y] = pval
    return tmax_pvals

def compute_pval(score, perm_scores):
    n_perm = len(perm_scores)
    pvalue = (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)
    return pvalue
