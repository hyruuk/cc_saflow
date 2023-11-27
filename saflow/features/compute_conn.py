import argparse
import saflow
import mne
from mne_bids import BIDSPath, read_raw_bids
import mne
import saflow
import os
import numpy as np
import pandas as pd
import mne_bids
from mne_bids import BIDSPath, read_raw_bids

import pickle
from saflow.features.utils import create_fnames, segment_sourcelevel

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='12',
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

