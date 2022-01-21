import os.path as op
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio

import ephypype
from ephypype.nodes import create_iterator
from ephypype.datasets import fetch_omega_dataset
from saflow import BIDS_PATH

#base_path = op.join(op.dirname(ephypype.__file__), '..', 'examples')
#data_path = fetch_omega_dataset(base_path)
data_path = op.join('/scratch/hyruuk/saflow_data/saflow_bids') # BIDS_PATH

#### PARAMETERS
import json  # noqa
import pprint  # noqa
params = json.load(open("params.json"))

pprint.pprint({'experiment parameters': params["general"]})
subject_ids = params["general"]["subject_ids"]  # sub-003
session_ids = params["general"]["session_ids"]
run_ids = params["general"]["run_ids"]  # ses-0001
NJOBS = params["general"]["NJOBS"]

pprint.pprint({'inverse parameters': params["inverse"]})
spacing = params["inverse"]['spacing']  # ico-5 vs oct-6
snr = params["inverse"]['snr']  # use smaller SNR for raw data
inv_method = params["inverse"]['img_method']  # sLORETA, MNE, dSPM, LCMV
parc = params["inverse"]['parcellation']  # parcellation to use: 'aparc' vs 'aparc.a2009s'  # noqa
# noise covariance matrix filename template
noise_cov_fname = params["inverse"]['noise_cov_fname']

# set sbj dir path, i.e. where the FS folfers are
subjects_dir = op.join(data_path, params["general"]["subjects_dir"])

########

# workflow directory within the `base_dir`
src_reconstruction_pipeline_name = 'source_reconstruction_' + \
    inv_method + '_' + parc.replace('.', '')

main_workflow = pe.Workflow(name=src_reconstruction_pipeline_name)
main_workflow.base_dir = data_path

infosource = create_iterator(['subject_id', 'session_id', 'run_id'],
                             [subject_ids, session_ids, run_ids])
############

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['raw_file', 'trans_file']),  # noqa
                     name='datasource')

datasource.inputs.base_directory = data_path
datasource.inputs.template = '*%s/%s/meg/%s_%s_task-gradCPT_%s_meg_%s.fif'

datasource.inputs.template_args = dict(
        raw_file=[['subject_id', 'session_id', 'subject_id', 'session_id', 'run_id', '-epo']],
        trans_file=[['subject_id', 'session_id', 'subject_id', 'session_id', 'run_id', '-epotrans']])

datasource.inputs.sort_filelist = True

###########
from ephypype.pipelines import create_pipeline_source_reconstruction  # noqa
event_id = {'Freq': 21, 'Rare': 31}
inv_sol_workflow = create_pipeline_source_reconstruction(
    data_path, subjects_dir, spacing=spacing, inv_method=inv_method, parc=parc,
    noise_cov_fname=noise_cov_fname, is_epoched=True, events_id={}, ROIs_mean=False, all_src_space=True)

###########

main_workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
main_workflow.connect(infosource, 'session_id', datasource, 'session_id')
main_workflow.connect(infosource, 'run_id', datasource, 'run_id')

##########

main_workflow.connect(infosource, 'subject_id',
                      inv_sol_workflow, 'inputnode.sbj_id')
main_workflow.connect(datasource, 'raw_file',
                      inv_sol_workflow, 'inputnode.raw')
main_workflow.connect(datasource, 'trans_file',
                      inv_sol_workflow, 'inputnode.trans_file')


##########
#main_workflow.write_graph(graph2use='colored')  # colored

#########
#import matplotlib.pyplot as plt  # noqa
#img = plt.imread(op.join(data_path, src_reconstruction_pipeline_name, 'graph.png'))  # noqa
#plt.figure(figsize=(8, 8))
#plt.imshow(img)
#plt.axis('off')

#########
main_workflow.config['execution'] = {'remove_unnecessary_outputs': 'false'}

# Run workflow locally on 1 CPU
main_workflow.run(plugin='MultiProc', plugin_args={'n_procs': NJOBS})
