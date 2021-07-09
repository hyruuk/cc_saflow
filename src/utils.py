import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_SAflow_bids(BIDS_PATH, subj, run, stage, cond=None):
    '''
    Constructs BIDS basename and filepath in the SAflow database format.
    '''
    if run == '1' or run == '8': # determine task based on run number
        task = 'RS'
    else:
        task = 'gradCPT'

    if not('report' in stage) and 'epo' in stage or 'raw' in stage: # determine extension based on stage
        extension = '.fif'
    elif 'sources' in stage or 'TFR' in stage:
        extension = '.hd5'
    elif 'events' in stage:
        extension = '.tsv'
    elif 'ARlog' in stage or 'PSD' in stage:
        extension = '.pkl'
    elif 'report' in stage:
        extension = '.html'

    if 'events' in stage:
        SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_{}{}'.format(subj, task, run, stage, extension)
    else:
        if cond == None: # build basename with or without cond
            SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_meg_{}{}'.format(subj, task, run, stage, extension)
        else:
            SAflow_bidsname = 'sub-{}_ses-recording_task-{}_run-0{}_meg_{}_{}{}'.format(subj, task, run, stage, cond, extension)

    SAflow_bidspath = os.path.join(BIDS_PATH, 'sub-{}'.format(subj), 'ses-recording', 'meg', SAflow_bidsname)
    return SAflow_bidsname, SAflow_bidspath



def array_topoplot(toplot, ch_xy, showtitle=False, titles=None, savefig=False, figpath=None, vmin=-1, vmax=1, cmap='magma', with_mask=False, masks=None, show=True):
    #create fig
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=5)
    fig, ax = plt.subplots(1,len(toplot), figsize=(20,10), squeeze=False)
    for i, data in enumerate(toplot):
        if with_mask == False:
            data = np.reshape(data, (-1,))
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False, contours=None, extrapolate='box', outlines='head')
        elif with_mask == True:
            data = data.reshape((-1,)) #C'est ici que ça bug si je reshape pas il est pas content
            image,_ = mne.viz.plot_topomap(data=data, pos=ch_xy, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax[i], show=False, contours=None, mask_params=mask_params, mask=masks[i], extrapolate='box', outlines='head')
        #option for title
        if showtitle == True:
            ax[i].set_title(titles[i], fontdict={'fontsize': 20, 'fontweight': 'heavy'})
    #add a colorbar at the end of the line (weird trick from https://www.martinos.org/mne/stable/auto_tutorials/stats-sensor-space/plot_stats_spatio_temporal_cluster_sensors.html#sphx-glr-auto-tutorials-stats-sensor-space-plot-stats-spatio-temporal-cluster-sensors-py)
    divider = make_axes_locatable(ax[-1])
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_colorbar.tick_params(labelsize=14)
    #save plot if specified
    if savefig == True:
        plt.savefig(figpath, dpi=300)
    if show == True:
        plt.show()
        plt.close(fig=fig)
    else:
        plt.close(fig=fig)
    return fig

def create_pval_mask(pvals, alpha=0.05):
    mask = np.zeros((len(pvals),), dtype='bool')
    for i, pval in enumerate(pvals):
        if pval <= alpha:
            mask[i] = True
    return mask

def get_ch_pos(epochs):
    ### Obtain actual sensor positions for plotting (keep only channels that are present in the data)
    new_ch_names = [s.strip('3105') for s in epochs.ch_names] # ajuste les noms de channels avant de comparer channels présents sur layout et data
    actual_ch_names = [s.strip('-') for s in new_ch_names] # me demande pas pk faut le faire en 2 temps, marche pas sinon
    reference_layout = mne.channels.find_layout(epochs.info) # obtain the CTF 275 layout based on the channels names
    reference_ch_names = reference_layout.names # let's just be very explicit in here...
    reference_pos = reference_layout.pos # again
    not_in_actual = [x for x in reference_ch_names if not x in actual_ch_names] # find chan names that are in layout but not in data

    # loop to get the indexes of chans to remove from the layout
    idx_to_del = []
    for i in range(len(not_in_actual)):
        idx_to_del.append(reference_ch_names.index(not_in_actual[i])) # get layout index of every chan name not in data
    reverted_idx_to_del = idx_to_del[::-1]

    # actually removes the chans (f*** code efficiency)
    list_ref_pos = list(reference_pos)
    for i in range(len(reverted_idx_to_del)):
        del list_ref_pos[reverted_idx_to_del[i]] # delete 'em
    new_ref_pos = np.array(list_ref_pos)

    ch_xy = new_ref_pos[:,0:2] # retain only the X and Y coordinates (0:2 veut dire "de 0 à 2", donc 0 et 1 car on compte pas 2)
    return ch_xy
