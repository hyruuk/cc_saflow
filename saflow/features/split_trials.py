##### OPEN PREPROC FILES AND SEGMENT THEM
from saflow.neuro import split_PSD_data, split_trials
from saflow.saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, FEAT_PATH, BIDS_PATH, LOGS_DIR
from saflow.utils import get_SAflow_bids
from scipy.io import savemat
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--split",
    default=[25, 75],
    type=int,
    nargs='+',
    help="Bounds of percentile split",
)
parser.add_argument(
    "-by",
    "--by",
    default="VTC",
    type=str,
    help="Choose the basis on which to split the data ('VTC' or 'odd')",
)

args = parser.parse_args()


if __name__ == "__main__":
    for subj in SUBJ_LIST:
        for run in BLOCS_LIST:
            CONDS_LIST = args.split
            by = args.by

            if by == 'VTC':
                INepochs, OUTepochs = split_trials(BIDS_PATH, LOGS_DIR, subj=subj, run=run, stage='PSD', by='VTC', lobound=CONDS_LIST[0], hibound=CONDS_LIST[1])
                INepochs_path, INepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='IN{}'.format(CONDS_LIST[0]))
                OUTepochs_path, OUTepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='OUT{}'.format(CONDS_LIST[1]))

                with open(INepochs_filename, 'wb') as fp:
                    pickle.dump(INepochs, fp)
                with open(OUTepochs_filename, 'wb') as fp:
                    pickle.dump(OUTepochs, fp)

            elif by == 'odd':
                FREQepochs, RAREepochs = split_trials(BIDS_PATH, LOGS_DIR, subj=subj, run=run, stage='PSD', by='odd', oddball='hits')
                FREQepochs_path, FREQepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='FREQhits')
                RAREepochs_path, RAREepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='RAREhits')

                with open(FREQepochs_filename, 'wb') as fp:
                    pickle.dump(FREQepochs, fp)
                with open(RAREepochs_filename, 'wb') as fp:
                    pickle.dump(RAREepochs, fp)


            elif by == 'resp':
                RESPepochs, NORESPepochs = split_trials(BIDS_PATH, LOGS_DIR, subj=subj, run=run, stage='PSD', by='resp')
                RESPepochs_path, RESPepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='RESP')
                NORESPepochs_path, NORESPepochs_filename = get_SAflow_bids(BIDS_PATH, subj=subj, run=run, stage='PSD', cond='NORESP')

                with open(RESPepochs_filename, 'wb') as fp:
                    pickle.dump(RESPepochs, fp)
                with open(NORESPepochs_filename, 'wb') as fp:
                    pickle.dump(NORESPepochs, fp)
