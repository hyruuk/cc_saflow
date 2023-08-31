import os
import os.path as op
#from saflow.utils import get_SAflow_bids

from saflow import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import argparse
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from autoreject import AutoReject
import mne
from mne.io import read_raw_ctf
from matplotlib.pyplot import close
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='05',
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-i",
    "--ica",
    default=True,
    type=bool,
    help="Preprocessing with or without ica"
)
args = parser.parse_args()

def saflow_preproc(filepath, ica=True):
    # Build output names
    report_path = str(input_path.copy().update(root=str(input_path.root) + '/derivatives/preprocessed/',
                                               description='report', 
                                               processing='clean', 
                                               suffix='meg')).replace('.ds', '.html')
    output_path = input_path.copy().update(root=str(input_path.root) + '/derivatives/preprocessed/',
                                               processing='clean', 
                                               suffix='meg',
                                               extension='.fif')
    # Load raw data
    report = mne.Report(verbose=True)
    raw_data = read_raw_bids(filepath, extra_params={'preload':True})
    raw_data = raw_data.apply_gradient_compensation(
        grade=3
    )  # required for source reconstruction
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude="bads")
    fig = raw_data.plot(show=False)
    report.add_figure(fig, title="Time series")
    #close(fig)
    fig = raw_data.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD")
    #close(fig)

    ## Filtering
    high_cutoff = 200
    low_cutoff = 0.5
    raw_data.filter(low_cutoff, high_cutoff, fir_design="firwin")
    raw_data.notch_filter(
        np.arange(60, high_cutoff + 1, 60),
        picks=picks,
        filter_length="auto",
        phase="zero",
        fir_design="firwin",
    )
    fig = raw_data.plot_psd(average=False, picks=picks, fmax=120, show=False)
    report.add_figure(fig, title="PSD filtered")
    #close(fig)
    if ica == False:
        write_raw_bids(raw_data, output_path, allow_preload=True, format='FIF', overwrite=True)
        report.save(report_path, open_browser=False, overwrite=True)
        del report
        del raw_data
        del fig


    elif ica == True:
        record_date = raw_data.info['meas_date'].strftime('%Y%m%d')
        noise_cov = load_noise_cov(record_date)
        ## ICA
        ica = ICA(n_components=20, 
                  random_state=0, 
                  noise_cov=noise_cov).fit(raw_data.copy().filter(1, None), decim=3)
        fig = ica.plot_sources(raw_data, show=False)
        report.add_figure(fig, title="Independent Components")
        #close(fig)

        ## FIND ECG COMPONENTS
        ecg_threshold = 0.50
        ecg_epochs = create_ecg_epochs(raw_data, ch_name="ECG")
        ecg_inds, ecg_scores = ica.find_bads_ecg(
            ecg_epochs, ch_name="ECG", method="ctps", threshold=ecg_threshold
        )
        fig = ica.plot_scores(ecg_scores, ecg_inds, show=False)
        report.add_figure(fig, title="Correlation with ECG")
        #close(fig)
        fig = list()
        try:
            fig = ica.plot_properties(
                ecg_epochs, picks=ecg_inds, image_args={"sigma": 1.0}, show=False
            )
            for i, figure in enumerate(fig):
                report.add_figure(figure, title="Detected component " + str(i))
                #close(figure)
        except:
            print("No component to remove")

        ## FIND EOG COMPONENTS
        eog_threshold = 4
        eog_epochs = create_eog_epochs(raw_data, ch_name="vEOG")
        eog_inds, eog_scores = ica.find_bads_eog(
            eog_epochs, ch_name="vEOG", threshold=eog_threshold
        )
        # TODO : if eog_inds == [] then eog_inds = [index(max(abs(eog_scores)))]
        fig = ica.plot_scores(eog_scores, eog_inds, show=False)
        report.add_figure(fig, title="Correlation with vEOG")
        #close(fig)
        fig = list()
        try:
            fig = ica.plot_properties(
                eog_epochs, picks=eog_inds, image_args={"sigma": 1.0}, show=False
            )
            for i, figure in enumerate(fig):
                report.add_figure(figure, title="Detected component " + str(i))
                #close(figure)
        except:
            print("No component to remove")

        ## EXCLUDE COMPONENTS
        ica.exclude = ecg_inds
        ica.apply(raw_data)
        ica.exclude = eog_inds
        ica.apply(raw_data)
        fig = raw_data.plot(show=False)
        # Plot the clean signal.
        report.add_figure(fig, title="After filtering + ICA")
        #close(fig)
        ## SAVE PREPROCESSED FILE
        write_raw_bids(raw_data, output_path, allow_preload=True, format='FIF', overwrite=True)
        report.save(report_path, open_browser=False, overwrite=True)
        del ica
        del report
        del raw_data
        del fig

def load_noise_cov(er_date, bids_root=BIDS_PATH):
    # Noise covariance matrix
    noise_cov_bidspath = BIDSPath(subject='emptyroom', 
                    session=er_date,
                    task='noise', 
                    datatype="meg",
                    processing='noisecov',
                    root=bids_root + '/derivatives/noise_cov/')
    noise_cov_fullpath = str(noise_cov_bidspath.fpath) + '.fif'
    if not os.path.isfile(noise_cov_fullpath):
        er_raw = read_raw_bids(BIDSPath(subject='emptyroom', 
                    session=er_date,
                    task='noise', 
                    datatype="meg",
                    root=bids_root + '/raw/'))
        os.makedirs(os.path.dirname(noise_cov_bidspath.fpath), exist_ok=True)
        noise_cov = mne.compute_raw_covariance(
                    er_raw, method=["shrunk", "empirical"], rank=None, verbose=True
                )
        noise_cov.save(noise_cov_bidspath)
    else:
        noise_cov = mne.read_cov(noise_cov_fullpath)
    return noise_cov


if __name__ == "__main__":
    #for subj in SUBJ_LIST:
    subj = args.subject
    ica = args.ica
    for bloc in BLOCS_LIST:
        input_path = BIDSPath(subject=subj,
                    task="gradCPT",
                    run='0'+bloc,
                    datatype="meg",
                    extension='.ds',
                    processing=None,
                    description=None,
                    root=BIDS_PATH)

                
        saflow_preproc(input_path, ica=ica)