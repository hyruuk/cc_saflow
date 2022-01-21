"""
.. _spectral_connectivity:

====================================
Compute connectivity on sensor space
====================================
The connectivity pipeline performs connectivity analysis in
sensor or source space.

The **input** data should be a time series matrix in **npy** or **mat** format.
"""

# Authors: Annalisa Pascarella <a.pascarella@iac.cnr.it>
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 2
import os.path as op
import numpy as np
import nipype.pipeline.engine as pe

import ephypype
from ephypype.nodes import create_iterator, create_datagrabber
from ephypype.nodes import get_frequency_band
from ephypype.datasets import fetch_omega_dataset
import nipype.interfaces.io as nio
import sys

###############################################################################
# Let us fetch the data first. It is around 675 MB download.

data_path = op.join('/scratch/hyruuk/saflow_data/saflow_bids')

###############################################################################
# then read the parameters for experiment and connectivity from a
# :download:`json <https://github.com/neuropycon/ephypype/blob/master/examples/params.json>`
# file and print it

import json  # noqa
import pprint  # noqa
params = json.load(open("params.json"))

pprint.pprint({'experiment parameters': params["general"]})
#subject_ids = params["general"]["subject_ids"]  # sub-003
subject_ids = sys.argv[1:][0]
subject_ids = ['sub-' + str(subject_ids)]
session_ids = params["general"]["session_ids"]  # ses-0001
run_ids = params["general"]["run_ids"]  # ses-0001
cond_ids = params["general"]["cond_ids"]
cond_ids = [str(sys.argv[1:][1])]
NJOBS = params["general"]["NJOBS"]

pprint.pprint({'connectivity parameters': params["connectivity"]})
freq_band_names = params["connectivity"]['freq_band_names']
freq_bands = params["connectivity"]['freq_bands']
con_method = params["connectivity"]['method']
epoch_window_length = params["connectivity"]['epoch_window_length']
sfreq = 1200

###############################################################################
# Then, we create our workflow and specify the `base_dir` which tells
# nipype the directory in which to store the outputs.

# workflow directory within the `base_dir`
correl_analysis_name = 'spectral_connectivity_' + con_method

main_workflow = pe.Workflow(name=correl_analysis_name)
main_workflow.base_dir = data_path

###############################################################################
# Then we create a node to pass input filenames to DataGrabber from nipype

infosource = create_iterator(['subject_id', 'session_id', 'run_id', 'cond_id', 'freq_band_name'],
                             [subject_ids, session_ids, run_ids, cond_ids, freq_band_names])

###############################################################################
# and a node to grab data. The template_args in this node iterate upon
# the values in the infosource node
sources_fp = '/scratch/hyruuk/saflow_data/saflow_bids/source_reconstruction_MNE_aparca2009s/inv_sol_pipeline/'
template_path = sources_fp + '_run_id_%s_session_id_%s_subject_id_%s/inv_solution/%s_%s_task-gradCPT_%s_meg_%s_-epo_stc.hdf5'
template_args = [['run_id', 'session_id', 'subject_id', 'subject_id', 'session_id', 'run_id', 'cond_id']]

datasource = pe.Node(
    interface=nio.DataGrabber(infields=['subject_id', 'session_id', 'run_id', 'cond_id'], outfields=['ts_file']),
    name='datasource')

datasource.inputs.base_directory = data_path
datasource.inputs.template = template_path

datasource.inputs.template_args = dict(ts_file=template_args)
datasource.inputs.sort_filelist = True

###############################################################################
# Ephypype creates for us a pipeline which can be connected to these
# nodes we created. The connectivity pipeline is implemented by the function
# :func:`ephypype.pipelines.ts_to_conmat.create_pipeline_time_series_to_spectral_connectivity`,
# thus to instantiate this connectivity pipeline node, we import it and pass
# our parameters to it.
# The connectivity pipeline contains two nodes and is based on the MNE Python
# functions computing frequency- and time-frequency-domain connectivity
# measures. A list of the different connectivity measures implemented by MNE
# can be found in the description of :func:`mne.viz.plot_connectivity_circle`
#
# In particular, the two nodes are:
#
# * :class:`ephypype.interfaces.mne.spectral.SpectralConn` computes spectral connectivity in a given frequency bands
# * :class:`ephypype.interfaces.mne.spectral.PlotSpectralConn` plot connectivity matrix using the |plot_connectivity_circle| function
#
# .. |plot_connectivity_circle| raw:: html
#
#   <a href="http://martinos.org/mne/stable/generated/mne.viz.plot_connectivity_circle.html#mne.viz.plot_connectivity_circle" target="_blank">spectral_connectivity function</a>

from ephypype.pipelines import create_pipeline_time_series_to_spectral_connectivity # noqa
spectral_workflow = create_pipeline_time_series_to_spectral_connectivity(
    data_path, con_method=con_method,
    epoch_window_length=None, is_sensor_space=False)

###############################################################################
# The connectivity node needs two auxiliary nodes: one node reads the raw data
# file in .fif format and extract the data and the channel information; the
# other node get information on the frequency band we are interested on.

frequency_node = get_frequency_band(freq_band_names, freq_bands)

###############################################################################
# We then connect the nodes two at a time. First, we connect two outputs
# (subject_id and session_id) of the infosource node to the datasource node.
# So, these two nodes taken together can grab data.
# The third output of infosource (freq_band_name) is connected to the
# frequency node

main_workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
main_workflow.connect(infosource, 'session_id', datasource, 'session_id')
main_workflow.connect(infosource, 'run_id', datasource, 'run_id')
main_workflow.connect(infosource, 'cond_id', datasource, 'cond_id')
main_workflow.connect(infosource, 'freq_band_name',
                      frequency_node, 'freq_band_name')

###############################################################################
# Similarly, for the inputnode of create_array_node and spectral_workflow.
# Things will become clearer in a moment when we plot the graph of the workflow

main_workflow.connect(datasource, 'ts_file',
                      spectral_workflow, 'inputnode.ts_file')

main_workflow.connect(frequency_node, 'freq_bands',
                      spectral_workflow, 'inputnode.freq_band')
spectral_workflow.inputs.inputnode.sfreq = sfreq

###############################################################################
# main_workflow.write_graph(graph2use='colored')  # colored

###############################################################################
# and visualize it. Take a moment to pause and notice how the connections
# here correspond to how we connected the nodes.

#import matplotlib.pyplot as plt  # noqa
#img = plt.imread(op.join(data_path, correl_analysis_name, 'graph.png'))
#plt.figure(figsize=(8, 8))
#plt.imshow(img)
#plt.axis('off')

###############################################################################
# Finally, we are now ready to execute our workflow.

main_workflow.config['execution'] = {'remove_unnecessary_outputs': 'false'}

# Run workflow locally on 1 CPU
main_workflow.run(plugin='MultiProc', plugin_args={'n_procs': NJOBS})
