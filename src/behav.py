from scipy.io import loadmat, savemat
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def find_logfile(subj, bloc, log_files):
    ### Find the right logfile for a specific subject and bloc in a list of log_files
    # (typically the list of files in the log folder, obtained by "os.listdir(LOGS_DIR)")
    '''
    Be careful, because the logfile number 1 corresponds to the recording number 2 etc...
    '''
    for file in log_files:
        if file[7:9] ==  subj and int(file[10]) == int(bloc)-1:
            break
    return file

def interp_RT(RT):
    ### Interpolate missing reaction times using the average of proximal values.
    # Note that this technique behaves poorly when two 0 are following each other
    for i in range(len(RT)):
        if RT[i] == 0:
            try:
                if RT[i+1] != 0 and RT[i-1] != 0:
                    RT[i] = np.mean((RT[i-1], RT[i+1]))
                else:
                    RT[i] = RT[i-1]

            except: #if RT[i+1] or RT[i-1] doesn't exists (beginning or end of file) replace by nearest RT
                try:
                    RT[i] = RT[i-1]
                except:
                    RT[i] = RT[i+1]
    RT_interpolated = RT
    return RT_interpolated

def compute_VTC(RT_interp, filt=True, filt_order=3, filt_cutoff=0.05):
    ### Compute the variance time course (VTC) of the array RT_interp
    VTC = (RT_interp - np.mean(RT_interp))/np.std(RT_interp)
    if filt == True:
        b, a = signal.butter(filt_order,filt_cutoff)
        VTC_filtered = signal.filtfilt(b, a, abs(VTC))
        VTC = VTC_filtered
    return VTC

def in_out_zone(VTC, lobound = None, hibound = None):
    ### Collects the indices of IN/OUT zone trials
    # lobound and hibound are values between 0 and 1 representing quantiles
    INzone = []
    OUTzone = []
    if lobound == None and hibound == None:
        VTC_med = np.median(VTC)
        for i, val in enumerate(VTC):
            if val < VTC_med:
                INzone.append(i)
            if val >= VTC_med:
                OUTzone.append(i)
    else:
        low = np.quantile(VTC, lobound)
        high = np.quantile(VTC, hibound)
        for i, val in enumerate(VTC):
            if val < low:
                INzone.append(i)
            if val >= high:
                OUTzone.append(i)
    INzone = np.asarray(INzone)
    OUTzone = np.asarray(OUTzone)
    return INzone, OUTzone

def find_jumps(array):
    ### Finds the jumps in an array containing ordered sequences
    jumps = []
    for i,_ in enumerate(array):
        try:
            if array[i+1] != array[i]+1:
                jumps.append(i)
        except:
            break
    return jumps

def find_bounds(array):
    ### Create a list of tuples, each containing the first and last values of every ordered sequences
    # contained in a 1D array
    jumps = find_jumps(array)
    bounds = []
    for i, jump in enumerate(jumps):
        if jump == jumps[0]:
            bounds.append(tuple([array[0], array[jump]]))
        else:
            bounds.append(tuple([array[jumps[i-1]+1], array[jump]]))
        if i == len(jumps)-1:
            bounds.append(tuple([array[jump+1], array[-1]]))
    return bounds

def get_VTC_from_file(filepath, lobound=None, hibound=None, filt=True, filt_order=3, filt_cutoff=0.05):
    data = loadmat(filepath)
    df_response = pd.DataFrame(data['response'])
    df_response = df_response[:-1]
    RT_array= np.asarray(df_response.loc[:,4])
    RT_interp = interp_RT(RT_array)
    VTC = compute_VTC(RT_interp, filt=filt, filt_order=filt_order, filt_cutoff=filt_cutoff)
    INzone, OUTzone = in_out_zone(VTC, lobound=lobound, hibound=hibound)
    INbounds = find_bounds(INzone)
    OUTbounds = find_bounds(OUTzone)
    return VTC, INbounds, OUTbounds, INzone, OUTzone, RT_array

def plot_VTC(VTC, figpath=None, save=False, INOUT=True):
    x = np.arange(0, len(VTC))
    if INOUT:
        OUT_mask = np.ma.masked_where(VTC >= np.median(VTC), VTC)
        IN_mask = np.ma.masked_where(VTC < np.median(VTC), VTC)
        lines = plt.plot(x, OUT_mask, x, IN_mask)
        fig = plt.plot()
        plt.setp(lines[0], linewidth=2)
        plt.setp(lines[1], linewidth=2)
        plt.legend(('IN zone', 'OUT zone'), loc='upper right')
        plt.title('IN vs OUT zone')
    else:
        line = plt.plot(x, VTC)
        fig = plt.plot()
        plt.setp(line, linewidth=2)
        plt.title('VTC')
    if save == True:
        plt.savefig(figpath)
    plt.show()

### SDT
from scipy.stats import norm
import math

Z = norm.ppf

def SDT(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
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
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    return out
