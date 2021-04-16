import os.path as op

BIDS_PATH = '/storage/Yann/saflow_DATA/saflow_bids'
ACQ_PATH = '/storage/Yann/saflow_DATA/acquisition'

FOLDERPATH = '/storage/Yann/saflow_DATA/saflow_bids/'
RESULTS_PATH = '/home/karim/pCloudDrive/science/saflow/results/single_feat/'
IMG_DIR = '/home/karim/pCloudDrive/science/saflow/images/'
CH_FILE = '/home/karim/pCloudDrive/science/saflow/sub-04_ses-recording_task-gradCPT_run-02_meg.ds'
LOGS_DIR = '/home/karim/pCloudDrive/science/saflow/gradCPT/gradCPT_share_Mac_PC/gradCPT_share_Mac_PC/saflow_data/'
REPORTS_PATH = op.join(FOLDERPATH, 'preproc_reports')
FEAT_PATH = FOLDERPATH + 'features/'
FREQS = [ [2, 4], [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
FREQS_NAMES = ['delta', 'theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
BLOCS_LIST = ['2','3', '4', '5', '6', '7']
ZONE_CONDS = ['IN', 'OUT']
ZONE2575_CONDS = ['IN25', 'OUT75']
FS_SUBJDIR = '/storage/Yann/saflow_DATA/saflow_anat/'
