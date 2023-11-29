import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from fooof import FOOOF, Bands, FOOOFGroup
import mne_bids
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def grid_topoplot(array_data, chan_info, titles_x, titles_y, masks=None, mask_params=None, cmap=None, vlims=None, title=None):
    '''Creates a grid of topoplots from the array_data. First dimension is used for the rows, second for the columns'''
    letters = ['A', 'B', 'C']
    if vlims is None:
        vlims = [(-np.max(abs(row_data)), np.max(abs(row_data))) for row_data in array_data]
        
    fig, axes = plt.subplots(array_data.shape[0], array_data.shape[1], figsize=(3*array_data.shape[1], 3*array_data.shape[0]))
    plt.subplots_adjust(wspace=0.1, hspace=0)
    for idx_row, row in enumerate(axes):
        for idx_col, ax in enumerate(row):
            mne.viz.plot_topomap(array_data[idx_row, idx_col], 
                                chan_info, 
                                axes=ax, 
                                show=False, 
                                cmap=cmap[idx_row] if cmap is not None else 'magma',
                                mask=masks[idx_row, idx_col] if masks is not None else None,
                                vlim=vlims[idx_row] if vlims is not None else None,
                                extrapolate="local",
                                outlines="head",
                                sphere=0.15,
                                contours=0,
            )
            if idx_row == 0:
                ax.set_title(titles_x[idx_col])
            if idx_col == 0:
                ax.set_ylabel(titles_y[idx_row], fontsize=14, rotation=0, labelpad=45)
                ax.text(
                        -0.02,
                        1.01,
                        letters[idx_row],
                        transform=ax.transAxes,
                        size=14,
                        weight="bold",
                    )
    # Add a colorbar and title. For this we need to use the figure handle.
    for row_idx in range(array_data.shape[0]):
        fig.colorbar(axes[row_idx][0].images[-1], ax=axes[row_idx], orientation='vertical', fraction=.005)

    if title is not None:
        fig.suptitle(title, y=1, fontsize=18)
    return fig, axes