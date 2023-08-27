### Check ISI for each bloc of each subject

from mne.io import read_raw_fif
from mne import find_events
from mne_bids import BIDSPath
from saflow import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import numpy as np
import pandas as pd

if __name__ == "__main__":
    isi_dict {'subject':[],
                'bloc':[],
                'ISI':[]}
    for subj in SUBJ_LIST:
        for bloc in ['2']:#BLOCS_LIST:
            raw_path = BIDSPath(subject=subj,
                                task="gradCPT",
                                run='0'+bloc,
                                datatype="meg",
                                processing=None,
                                description=None,
                                extension=".fif",
                                root=BIDS_PATH)
            raw = read_raw_fif(raw_path, preload=False, verbose=False)
            events = find_events(raw, min_duration=2 / raw.info["sfreq"], verbose=False)
            events_cleaned = np.array([x for x in events if x[2] != 99])
            average_isi = np.mean(np.diff(events_cleaned[:,0]))
            print(f'sub-{subj} bloc-{bloc} ISI: {average_isi}')
            isi_dict['subject'].append(subj)
            isi_dict['bloc'].append(bloc)
            isi_dict['ISI'].append(average_isi)
    isi_df = pd.DataFrame.from_dict(isi_dict)
    #isi_df.to_csv('ISI.csv')