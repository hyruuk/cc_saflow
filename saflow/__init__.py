import os.path as op

DATA_ROOT = '/media/hyruuk/YH_storage/DATA/saflow/'

BIDS_PATH = op.join(DATA_ROOT, 'bids')
ACQ_PATH = op.join(DATA_ROOT, 'sourcedata')
LOGS_DIR = op.join(DATA_ROOT, 'sourcedata', 'behav')

RESULTS_PATH = './results/'
IMG_DIR = './reports/figures/'

FEAT_PATH = './features/'
FREQS = [ [2, 4], [4, 8], [8, 12], [12, 20], [20, 30], [30, 60], [60, 90], [90, 120] ]
FREQS_NAMES = ['delta', 'theta', 'alpha', 'lobeta', 'hibeta', 'gamma1', 'gamma2', 'gamma3']
SUBJ_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18', '19', '20', '21', '22', '23', '24', '26', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']
BLOCS_LIST = ['2','3', '4', '5', '6', '7']
ZONE_CONDS = ['IN', 'OUT']
ZONE2575_CONDS = ['IN25', 'OUT75']
FS_SUBJDIR = op.join(DATA_ROOT, 'fs_subjects')
