# This code uses recon-all from FreeSurfer to perform skull stripping and brain segmentation on T1w MRI images
# It accepts a data directory as input and outputs a directory with the processed images
# The code assumes that the data directory contains subfolders with raw DICOM images
# The code will create a subfolder in the data directory for each subject with the processed images

import argparse
import os
import os.path as op

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--datapath",
    default="./",
    type=str,
    help="Path to raw DICOM images",
)
parser.add_argument(
    "-fd",
    "--fsdatapath",
    default="/home/hyruuk/freesurfer/subjects",
    type=str,
    help="Path to FS SUBJ_DIR",
)

args = parser.parse_args()

def main():
    path_to_data = args.datapath
    fs_datapath = args.fsdatapath

    for foldername in os.listdir(path_to_data):
        if "anat" in foldername:
            if not ".tar" in foldername:
                sub = "sub-" + foldername.split("_")[1]

                subfolders_list = sorted(os.listdir(op.join(path_to_data, foldername)))

                for subfolder in subfolders_list:
                    if "0.8mm_T1w" in subfolder:
                        images_fname = sorted(os.listdir(op.join(path_to_data, foldername, subfolder)))
                        first_frame = op.join(path_to_data, foldername, subfolder, images_fname[0])
                        print(first_frame)
                        print(subfolder)

                        os.system(f"recon-all -i {first_frame} -subjid {sub}")
                        os.system(f"recon-all -all -subjid {sub}")
                        break


if __name__ == "__main__":
    main()