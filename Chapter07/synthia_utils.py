"""
File name: tf_losses_and_metrics.py
Author: Benjamin Planche
Date created: 14.02.2019
Date last modified: 14:49 14.02.2019
Python Version: 3.6

Copyright = "Copyright (C) 2018-2019 of Packt"
Credits = ["Eliot Andres, Benjamin Planche"]
License = "MIT"
Version = "1.0.0"
Maintainer = "non"
Status = "Prototype" # "Prototype", "Development", or "Production"
"""

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import cityscapesscripts.helpers.labels as cityscapes_labels
import glob
import numpy as np
import functools
import cv2  # pip install opencv-python
from plot_utils import plot_image_grid

# As we use Synthia along Cityscapes, the other constants and functions can be found in `cityscapes_utils.py`


#==============================================================================
# Constant Definitions
#==============================================================================

SYNTHIA_FOLDER        = os.getenv('SYNTHIA_DATASET', default=os.path.expanduser('~/datasets/synthia'))

# Synthia file template:
SYNTHIA_FILE_TEMPLATE = os.path.join(
    '{root}', '{seq}', '{type}', '{type2}', '{direction}', '{frame:{filler}>{len_fill}}{ext}')
# (typical path is e.g. "/path/to/synthia/SYNTHIA-SEQS-XX/RGB/Stereo_Left/Omni_F/000000.png")

# Synthia "Rand Cityscapes" file template :
SYNTHIA_RAND_CS_FILE_TEMPLATE = os.path.join(
    '{root}', '{seq}', '{type}', '{frame:{filler}>{len_fill}}{ext}')
# (typical path is e.g. "/path/to/synthia/RAND_CITYSCAPES/RGB/000000.png")

# Synthia to Cityscapes label mapping:
SYNTHIA_2_CITYSCAPES_MAPPING = [
    [ 0, 255], # void
    [ 1,  10], # sky
    [ 2,   2], # building
    [ 3,   0], # road
    [ 4,   1], # sidewalk
    [ 5,   4], # fence
    [ 6,   8], # vegetation
    [ 7,   5], # pole
    [ 8,  13], # car
    [ 9,   7], # traffic sign
    [10,  11], # pedestrian/person
    [11,  18], # bicycle
    [12,   0], # lanemarking/road
    [13, 255], # parking-slot
    [14, 255], # road-work
    [15,   6]  # traffic light
               # other Cityscapes classes are missing from Synthia (rider,truck, ...)
]

# Synthia "Rand Cityscapes" to Cityscapes label mapping:
SYNTHIA_RAND_CS_2_CITYSCAPES_MAPPING = [
    [ 0, 255], # void
    [ 1,  10], # sky
    [ 2,   2], # building
    [ 3,   0], # road
    [ 4,   1], # sidewalk
    [ 5,   4], # fence
    [ 6,   8], # vegetation
    [ 7,   5], # pole
    [ 8,  13], # car
    [ 9,   7], # traffic sign
    [10,  11], # pedestrian/person
    [11,  18], # bicycle
    [12,  17], # motorcycle   (in other Synthia sequences, 12 is lanemarking)
    [13, 255], # parking-slot
    [14, 255], # road-work
    [15,   6], # traffic light
    [16,   9], # terrain      (only for Synthia "Rand Cityscapes")
    [17,  12], # rider        (only for Synthia "Rand Cityscapes")
    [18,  14], # truck        (only for Synthia "Rand Cityscapes")
    [19,  15], # bus          (only for Synthia "Rand Cityscapes")
    [20,  16], # train        (only for Synthia "Rand Cityscapes")
    [21,   3], # wall         (only for Synthia "Rand Cityscapes")
    [22,   0]  # lanemarking/road
               # other Cityscapes classes are missing from Synthia (rider,truck, ...)
]

#==============================================================================
# Function Definitions
#==============================================================================


def get_synthia_file_pairs(sequence='*', frame='*', ext='.png',
                           direction='*', type2='*', gt_type='LABELS',
                           root_folder=SYNTHIA_FOLDER, file_template=SYNTHIA_FILE_TEMPLATE):
    """
    Fetch pairs of filenames for the Synthia dataset.
    Note: wildcards accepted for the parameters (e.g. city='*' to return image pairs from every city)
    :param sequence:        Name of the sequence(s)
    :param frame:           Name of the frame
    :param ext:             File extension
    :param direction:       Camera side ("Omni_F" for face, "Omni_B" for back, "Omni_L" for left, ...)
    :param type2:           'Stereo_Left' or 'Stereo_Right'
    :param gt_type:         Type of GT ("COLORS" or "LABELS" by default)
    :param root_folder:     Synthia root folder
    :param file_template:   File template to be applied (default corresponds to Synthia original format)
    :return:                List of input files, List of corresponding GT files
    """
    input_file_template = file_template.format(
        root=root_folder, type='RGB', type2=type2, direction=direction,
        len_fill=1, filler='*', seq=sequence, frame=frame, ext=ext)
    input_files = glob.glob(input_file_template)

    gt_file_template = file_template.format(
        root=root_folder, type=os.path.join('GT', gt_type), type2=type2, direction=direction,
        len_fill=1, filler='*', seq=sequence, frame=frame, ext=ext)
    gt_files = glob.glob(gt_file_template)

    assert (len(input_files) == len(gt_files))
    return sorted(input_files), sorted(gt_files)


def convert_synthia_labels_to_cityscapes_format(
        gt_files, gt_label_folder=os.path.join('GT', 'LABEL_CS'), mapping=SYNTHIA_2_CITYSCAPES_MAPPING):
    """
    Convert and save Synthia semantic labels into a format conform to Cityscapes data.
    :param gt_files:            List of Synthia label files to convert
    :param gt_label_folder:     Folder where to save the converted labels, relative to Synthia root folder.
    :param mapping:             List mapping Synthia labels to Cityscapes one (see constants in this module)
    :return:                    /
    """
    for gt_file in gt_files:
        gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
        # Keep only the last channel (1st is unsused, 2nd is for instance segmentation):
        gt = gt[..., 2]
        # Map Synthia class values to Cityscapes ones:
        gt_cs = np.copy(gt)
        for i in range(len(mapping)):
            # Get mask:
            mask = gt == mapping[i][0]
            # Replace:
            gt_cs[mask] = mapping[i][1]

        gt_cs_file = gt_file.replace(os.path.join('GT', 'LABELS'), gt_label_folder)
        gt_cs_dir = os.path.dirname(gt_cs_file)
        if not os.path.exists(gt_cs_dir):
            os.makedirs(gt_cs_dir)
        cv2.imwrite(gt_cs_file, gt_cs.astype(np.uint8))
    return