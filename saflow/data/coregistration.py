import numpy as np
import mne
from mne.coreg import Coregistration
from mne.io import read_info
from mne_bids import BIDSPath

#data_path = mne.datasets.sample.data_path()
# data_path and all paths built from it are pathlib.Path objects
subjects_dir = "/home/hyruuk/freesurfer/subjects"
subject = "04"
task = "gradCPT"
run = "03"

bids_root = "/media/hyruuk/CoCoLabYANN/coco_data/saflow/bids"

bidspath = BIDSPath(subject=subject,
                    task=task,
                    run=run,
                    datatype="meg",
                    extension=".fif",
                    root=bids_root)

#fname_raw = data_path / "MEG" / subject / f"{subject}_audvis_raw.fif"
info = read_info(bidspath)
plot_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))


# Setup coreg model
fiducials = "auto"  # get fiducials from subject digitization
coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)