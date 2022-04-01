from scipy.io import loadmat, savemat
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os.path as op
from saflow import LOGS_DIR


def find_logfile(subj, bloc, log_files):
    ### Find the right logfile for a specific subject and bloc in a list of log_files
    # (typically the list of files in the log folder, obtained by "os.listdir(LOGS_DIR)")
    """
    Be careful, because the logfile number 1 corresponds to the recording number 2 etc...
    """
    for file in log_files:
        if file[7:9] == subj and int(file[10]) == int(bloc) - 1:
            break
    return file


def interpolate_RT(RT_raw):
    """Interpolates missing reaction times from the two nearest RTs.

    Parameters
    ----------
    RT_raw : np.array
        Raw reaction times as floats, with 0 for missing RT.

    Returns
    -------
    np.array
        The same array, but with 0s being replaced by the average of the two
        nearest RTs.

    """
    RT_array = RT_raw.copy()
    for idx, val in enumerate(RT_array):
        if val == 0:
            idx_next_val = 1
            try:
                while RT_array[idx + idx_next_val] == 0:  # Find next non-zero value
                    idx_next_val += 1
                if idx == 0:  # If first value is zero, use the next non-zero value
                    RT_array[idx] = RT_array[idx + idx_next_val]
                else:  # else use the average of the two nearest non-zero
                    RT_array[idx] = RT_array[idx - 1] + RT_array[idx + idx_next_val]
            except IndexError:  # If end of file is reached, use the last non-zero
                RT_array[idx] = RT_array[idx - 1]
    return RT_array


def compute_VTC(RT_array, subj_mean, subj_std):
    """Computes the raw (unfiltered) VTC.

    Parameters
    ----------
    RT_array : np.array
        Array of reaction times after interpolation.
    subj_mean : float
        Mean reaction time of a subject across all runs.
    subj_std : float
        Standard deviation of reaction times of a subject across all runs.

    Returns
    -------
    np.array
        Array containing VTC values, should be the same length as the RT array.

    """
    return abs((RT_array - subj_mean) / subj_std)


def clean_comerr(df_response):
    cleaned_df = df_response.copy()
    correct_omission_idx = []
    commission_error_idx = []
    correct_commission_idx = []
    omission_error_idx = []
    for idx_line, line in enumerate(cleaned_df.iterrows()):
        if line[1][0] == 1.0 and line[1][1] != 0.0:  # Rare stim with response
            cleaned_df.loc[idx_line, 4] == 0.0
            commission_error_idx.append(idx_line)
        if line[1][0] == 1.0 and line[1][1] == 0.0:  # Rare stim without response
            correct_omission_idx.append(idx_line)
        if line[1][0] == 2.0 and line[1][1] != 0.0:  # Freq stim with response
            correct_commission_idx.append(idx_line)
        if line[1][0] == 2.0 and line[1][1] == 0.0:  # Freq stim without response
            omission_error_idx.append(idx_line)

    performance_dict = {
        "commission_error": commission_error_idx,
        "correct_omission": correct_omission_idx,
        "omission_error": omission_error_idx,
        "correct_commission": correct_commission_idx,
    }
    return cleaned_df, performance_dict


def threshold_VTC(VTC, thresh=5):
    VTC[VTC >= thresh] = thresh
    return VTC


def get_VTC_from_file(
    subject,
    run,
    files_list,
    cpt_blocs=[2, 3, 4, 5, 6, 7],
    inout_bounds=[25, 75],
    filt_cutoff=0.05,
    filt_type="butterworth",
):
    """Short summary.

    Parameters
    ----------
    subject : type
        Description of parameter `subject`.
    run : type
        Description of parameter `run`.
    files_list : type
        Description of parameter `files_list`.
    cpt_blocs : type
        Description of parameter `cpt_blocs`.
    inout_bounds : type
        Description of parameter `inout_bounds`.

    Returns
    -------
    type
        Description of returned object.

    """
    # Find the logfiles belonging to a subject
    subject_logfiles = []
    for bloc in cpt_blocs:
        subject_logfiles.append(
            op.join(LOGS_DIR, find_logfile(subject, bloc, files_list))
        )

    # Load and clean RT arrays
    RT_arrays = []
    RT_to_VTC = []
    for idx_file, logfile in enumerate(subject_logfiles):
        data = loadmat(logfile)
        df_response = pd.DataFrame(data["response"])

        # Replace commission errors by 0
        df_clean, perf_dict = clean_comerr(df_response)
        RT_raw = np.asarray(df_clean.loc[:, 4])
        RT_raw = np.array([x if x != 0 else np.nan for x in RT_raw])  # zeros to nans
        # RT_interpolated = interpolate_RT(RT_raw)
        RT_arrays.append(RT_raw)
        if int(cpt_blocs[idx_file]) == int(run):
            RT_to_VTC = RT_raw
            performance_dict = perf_dict.copy()
            df_response_out = df_response

    # Obtain meand and std across runs
    allruns_RT_array = np.concatenate(RT_arrays)
    subj_mean = np.nanmean(allruns_RT_array)
    subj_std = np.nanstd(allruns_RT_array)

    # New VTC
    VTC_raw = compute_VTC(RT_to_VTC, subj_mean, subj_std)
    # VTC_thresholded = threshold_VTC(VTC_raw, thresh=3)  # Compute VTC remove variability values above threshold
    VTC_raw[np.isnan(VTC_raw)] = 0
    VTC_interpolated = interpolate_RT(VTC_raw)
    if filt_type == "gaussian":
        filt = signal.gaussian(9, 1)
        VTC_filtered = np.convolve(VTC_interpolated, filt)
    elif filt_type == "butterworth":
        b, a = signal.butter(3, filt_cutoff)  # (filt_order,filt_cutoff)
        VTC_filtered = signal.filtfilt(b, a, VTC_interpolated)

    IN_mask = np.ma.masked_where(
        VTC_filtered >= np.quantile(VTC_filtered, inout_bounds[0] / 100), VTC_filtered
    )
    OUT_mask = np.ma.masked_where(
        VTC_filtered < np.quantile(VTC_filtered, inout_bounds[1] / 100), VTC_filtered
    )
    IN_idx = np.where(IN_mask.mask == False)[0]
    OUT_idx = np.where(OUT_mask.mask == False)[0]

    return (
        IN_idx,
        OUT_idx,
        VTC_raw,
        VTC_filtered,
        IN_mask,
        OUT_mask,
        performance_dict,
        df_response_out,
    )


def plot_VTC(VTC_filtered, VTC_raw, IN_mask, OUT_mask, subject="?", bloc="?"):
    x = np.arange(0, len(VTC_raw))
    fig = plt.figure()
    raw = plt.plot(x, VTC_raw)
    plt.setp(raw, linewidth=0.5, color="black")
    lines = plt.plot(x, IN_mask, x, OUT_mask)
    plt.setp(lines[0], linewidth=2, color="blue")
    plt.setp(lines[1], linewidth=2, color="orange")

    plt.legend(("VTC", "IN zone", "OUT zone"), loc="upper right")
    plt.title(f"VTC plot (sub-{subject}, run-0{bloc})")

    # TODO : add lapses and correct target detection

    return fig


def compute_VTC(RT_array, subj_mean, subj_std):
    """Computes the raw (unfiltered) VTC.

    Parameters
    ----------
    RT_array : np.array
        Array of reaction times after interpolation.
    subj_mean : float
        Mean reaction time of a subject across all runs.
    subj_std : float
        Standard deviation of reaction times of a subject across all runs.

    Returns
    -------
    np.array
        Array containing VTC values, should be the same length as the RT array.

    """
    return abs((RT_array - subj_mean) / subj_std)


def old_plot_VTC(VTC, figpath=None, save=False, INOUT=True):
    x = np.arange(0, len(VTC))
    if INOUT:
        OUT_mask = np.ma.masked_where(VTC >= np.median(VTC), VTC)
        IN_mask = np.ma.masked_where(VTC <= np.median(VTC), VTC)
        lines = plt.plot(x, OUT_mask, x, IN_mask)
        fig = plt.plot()
        plt.setp(lines[0], linewidth=2)
        plt.setp(lines[1], linewidth=2)
        plt.legend(("IN zone", "OUT zone"), loc="upper right")
        plt.title("IN vs OUT zone")
    else:
        line = plt.plot(x, VTC)
        fig = plt.plot()
        plt.setp(line, linewidth=2)
        plt.title("VTC")
    if save == True:
        plt.savefig(figpath)
    plt.show()


### SDT
from scipy.stats import norm
import math

Z = norm.ppf


def SDT(hits, misses, fas, crs):
    """returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa
    # Return d', beta, c and Ad'
    out = {}
    out["d"] = Z(hit_rate) - Z(fa_rate)
    out["beta"] = math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)
    out["c"] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out["Ad"] = norm.cdf(out["d"] / math.sqrt(2))
    return out
