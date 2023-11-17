import saflow
from mne_bids import BIDSPath
import pickle

bids_root = saflow.BIDS_PATH
for subject in saflow.SUBJ_LIST:
    for bloc in saflow.BLOCS_LIST:
        try:
            run = '0' + bloc
            ARlog_bidspath = BIDSPath(subject=subject, 
                                    task='gradCPT', 
                                    run=run, 
                                    datatype='meg',
                                    suffix='meg',
                                    description='ARlog',
                                    root=bids_root + '/derivatives/preprocessed/')
            ARlog_bidspath = str(ARlog_bidspath.fpath)+'.pkl'
            with open(ARlog_bidspath, 'rb') as f:
                ARlog = pickle.load(f)
            print(f'sub-{subject} bloc-{run} : {sum(ARlog.bad_epochs)} bad epochs detected')
        except:
            print(f'No ARlog for sub-{subject} bloc-{run}')