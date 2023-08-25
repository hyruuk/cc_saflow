import os
import os.path as op
#from saflow.utils import get_SAflow_bids

from saflow import BIDS_PATH, SUBJ_LIST, BLOCS_LIST
import argparse
from mne_bids import BIDSPath
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from autoreject import AutoReject
import mne
from mne.io import read_raw_fif
from matplotlib.pyplot import close
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='04',
    type=str,
    help="Subject to process",
)
parser.add_argument(
	"-i",
	"--ica",
	default = True,
	type = bool,
	help="Preprocessing with or without ica"
)
args = parser.parse_args()

def saflow_preproc(filepath, savepath, reportpath, ica=True):
    report = mne.Report(verbose=True)
    raw_data = read_raw_fif(filepath, preload=True)
    raw_data = raw_data.apply_gradient_compensation(
        grade=3
    )  # required for source reconstruction
    picks = mne.pick_types(raw_data.info, meg=True, eog=True, exclude="bads")
    fig = raw_data.plot(show=False)
    report.add_figure(fig, title="Time series")
    close(fig)
    fig = raw_data.plot_psd(average=False, picks=picks, show=False)
    report.add_figure(fig, title="PSD")
    close(fig)

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
    close(fig)
    if ica == False:
        report.save(reportpath, open_browser=False, overwrite=True)
        raw_data.save(savepath, overwrite=True)
        del report
        del raw_data
        del fig

    elif ica == True:
        ## ICA
        ica = ICA(n_components=20, random_state=0).fit(raw_data, decim=3)
        fig = ica.plot_sources(raw_data, show=False)
        report.add_figure(fig, title="Independent Components")
        close(fig)

        ## FIND ECG COMPONENTS
        ecg_threshold = 0.50
        ecg_epochs = create_ecg_epochs(raw_data, ch_name="ECG")
        ecg_inds, ecg_scores = ica.find_bads_ecg(
            ecg_epochs, ch_name="ECG", method="ctps", threshold=ecg_threshold
        )
        fig = ica.plot_scores(ecg_scores, ecg_inds, show=False)
        report.add_figure(fig, title="Correlation with ECG")
        close(fig)
        fig = list()
        try:
            fig = ica.plot_properties(
                ecg_epochs, picks=ecg_inds, image_args={"sigma": 1.0}, show=False
            )
            for i, figure in enumerate(fig):
                report.add_figure(figure, title="Detected component " + str(i))
                close(figure)
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
        close(fig)
        fig = list()
        try:
            fig = ica.plot_properties(
                eog_epochs, picks=eog_inds, image_args={"sigma": 1.0}, show=False
            )
            for i, figure in enumerate(fig):
                report.add_figure(figure, title="Detected component " + str(i))
                close(figure)
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
        close(fig)
        ## SAVE PREPROCESSED FILE
        report.save(reportpath, open_browser=False, overwrite=True)
        raw_data.save(savepath, overwrite=True)
        del ica
        del report
        del raw_data
        del fig

if __name__ == "__main__":
	#for subj in SUBJ_LIST:
	subj = args.subject
	ica = args.ica
	for bloc in BLOCS_LIST:
		#file_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='raw')[1]
		#save_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_raw')[1]
		#report_path = get_SAflow_bids(BIDS_PATH, subj=subj, run=bloc, stage='preproc_report')[1]

		input_path = BIDSPath(subject=subj,
					task="gradCPT",
					run='0'+bloc,
					datatype="meg",
					extension=".fif",
					root=BIDS_PATH)
		output_path = BIDSPath(subject=subj,
			task="gradCPT",
			run='0'+bloc,
			datatype="meg",
			description='cleaned',
			extension=".fif",
			root=BIDS_PATH)
		report_path = str(input_path.fpath).replace('meg.fif', 'report.html')
		output_path = str(output_path.fpath) + '.fif'
		saflow_preproc(input_path, output_path, report_path, ica=ica)
