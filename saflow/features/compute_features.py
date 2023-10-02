import argparse
import saflow
import mne
from mne_bids import BIDSPath, read_raw_bids

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='23',
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-r",
    "--run",
    default='02',
    type=str,
    help="Run to process",
)

def create_fnames(subject, run, bids_path=saflow.BIDS_PATH):
    morph_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=run,
                            datatype='meg',
                            processing='clean',
                            description='morphed',
                            root=bids_path + '/derivatives/morphed_sources/')
    
    
    return {'morph':morph_bidspath,
            }

def compute_PSD():
    return

def compute_LZC(): # discretization='median' or 'perm'. If 'perm', k must be specified (start with k=3)
    return

def compute_slope():
    return

def compute_conmat(): # metric='wPLI' or 'granger'
    return

def segment_sources():
    return

if __name__ == "__main__":
    args = parser.parse_args()
    subject = args.subject
    run = args.run
    freq_bands = saflow.FREQS
    freq_names = saflow.FREQS_NAMES

    filepaths = create_fnames(subject, run)

    # Compute Hilbert