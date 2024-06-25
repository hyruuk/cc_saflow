from saflow import FS_SUBJDIR, SUBJ_LIST, BLOCS_LIST, BIDS_PATH
from mne_bids import BIDSPath
import mne
import numpy as np
import os.path as op
import os
from mne.datasets import fetch_fsaverage
from mne import Label
from collections import defaultdict
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default='37',
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
parser.add_argument(
    "-i",
    "--input",
    default='morphed_sources',
    type=str,
    help="Input state to use for source reconstruction",
)
parser.add_argument(
    "-a",
    "--atlas",
    default='aparc.a2009s',
    type=str,
    help="Atlas to use for average",
)
parser.add_argument(
    "-p",
    "--processing",
    default='clean',
    type=str,
    help="Processing state to use",

)

def create_fnames(subject, bloc, input, output, processing='clean'):
    input_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing=processing,
                            description='morphed',
                            root=BIDS_PATH + f'/derivatives/{input}/')
    input_bidspath.mkdir(exist_ok=True)
    output_bidspath = BIDSPath(subject=subject,
                            task='gradCPT',
                            run=bloc,
                            datatype='meg',
                            processing=processing,
                            description='atlased',
                            root=BIDS_PATH + f'/derivatives/{output}/')
    output_bidspath.mkdir(exist_ok=True)
    return {'input': input_bidspath, 'output': output_bidspath}


if __name__ == "__main__":
    args = parser.parse_args()
    subj = args.subject
    run = args.run
    input_state = args.input
    atlas = args.atlas
    processing = args.processing
    output_state = f'{input_state}_{atlas}'
    fnames = create_fnames(subj, run, input_state, output_state, processing)

    stc = mne.read_source_estimate(str(fnames['input'].fpath)+'-stc.h5')

    labels = mne.read_labels_from_annot('fsaverage', parc=atlas, subjects_dir=FS_SUBJDIR)

    # Initialize dictionary to store data for each region
    region_data = defaultdict(list)

    # Get the vertices for the left and right hemispheres from the SourceEstimate
    vertices_lh = stc.vertices[0]
    vertices_rh = stc.vertices[1]

    # Combine the vertex mappings into a dictionary
    vertex_to_region = {}

    # Map vertices to regions using the labels
    for label in labels:
        label_vertices = label.vertices
        hemi = 0 if label.hemi == 'lh' else 1
        if hemi == 0:
            common_vertices = np.intersect1d(vertices_lh, label_vertices)
        else:
            common_vertices = np.intersect1d(vertices_rh, label_vertices)
        
        for vert in common_vertices:
            vertex_to_region[vert] = label.name
 
    # Collect data for each region based on the vertex to region mapping
    for vert_idx, region in vertex_to_region.items():
        if vert_idx in vertices_lh:
            idx = np.where(vertices_lh == vert_idx)[0][0]
            region_data[region].append(stc.data[idx])
        elif vert_idx in vertices_rh:
            idx = np.where(vertices_rh == vert_idx)[0][0]
            region_data[region].append(stc.data[len(vertices_lh) + idx])

    # Average the data within each region
    region_averages = {region: np.mean(np.array(data), axis=0) for region, data in region_data.items()}

    # Convert dict to np array + list of names
    region_data = np.array(list(region_averages.values()))
    region_names = list(region_averages.keys())

    # Save atlased (region_averages) data with pickle
    with open(str(fnames['output'].fpath)+'-avg.pkl', 'wb') as f:
        pickle.dump({'data':region_data,
                     'region_names':region_names}, f)

