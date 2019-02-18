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

#==============================================================================
# Constant Definitions
#==============================================================================

CITYSCAPES_FOLDER = os.path.expanduser('~/datasets/cityscapes')

CITYSCAPES_IGNORE_VALUE     = 255
CITYSCAPES_LABELS           = [label for label in cityscapes_labels.labels
                               if -1 < label.trainId < CITYSCAPES_IGNORE_VALUE]
CITYSCAPES_COLORS           = np.asarray([label.color for label in CITYSCAPES_LABELS])
CITYSCAPES_COLORS_TF        = tf.constant(CITYSCAPES_COLORS, dtype=tf.int32)
CITYSCAPES_IMG_RATIO        = 2
CITYSCAPES_INT_FILL         = 6
CITYSCAPES_FILE_TEMPLATE    = os.path.join(
    '{root}', '{type}', '{split}', '{city}',
    '{city}_{seq:{filler}>{len_fill}}_{frame:{filler}>{len_fill}}_{type}{type2}{ext}')

#==============================================================================
# Function Definitions
#==============================================================================

# -----------------------------------------------------------------------------
#  DATA FUNCTIONS
# -----------------------------------------------------------------------------

def get_cityscapes_file_pairs(split='train', city='*', sequence='*', 
                              frame='*', ext='.png', gt_type='labelTrainIds',
                              root_folder=CITYSCAPES_FOLDER, file_template=CITYSCAPES_FILE_TEMPLATE):
    """
    Fetch pairs of filenames for the Cityscapes dataset.
    Note: wildcards accepted for the parameters (e.g. city='*' to return image pairs from every city)
    :param split:           Name of the split to return pairs from ("train", "val", ...)
    :param city:            Name of the city(ies)
    :param sequence:        Name of the video sequence(s)
    :param frame:           Name of the frame
    :param ext:             File extension
    :param gt_type:         Cityscapes GT type
    :param root_folder:     Cityscapes root folder
    :param file_template:   File template to be applied (default corresponds to Cityscapes original format)
    :return:                List of input files, List of corresponding GT files
    """
    input_file_template = file_template.format(
        root=root_folder, type='leftImg8bit', type2='', len_fill=1, filler='*',
        split=split, city=city, seq=sequence, frame=frame, ext=ext)
    input_files = glob.glob(input_file_template)
    
    gt_file_template = file_template.format(
        root=root_folder, type='gtFine', type2='_'+gt_type, len_fill=1, filler='*',
        split=split, city=city, seq=sequence, frame=frame, ext=ext)
    gt_files = glob.glob(gt_file_template)
    
    assert(len(input_files) == len(gt_files))
    
    return sorted(input_files), sorted(gt_files)


def _augmentation_fn(image, gt_image):
    """
    Apply random transformations to augment the training images.
    :param images:      Images
    :return:            Augmented Images
    """

    # If we decide to randomly flip or resize/crop the image, the same should be applied to
    # the label one so they still match. Therefore, to simplify the procedure, we stack the
    # two images together along the channel axis, before these random operations:
    stacked_images = tf.concat([image, tf.cast(gt_image, dtype=image.dtype)], axis=-1)

    # Randomly applied horizontal flip:
    stacked_images = tf.image.random_flip_left_right(stacked_images)

    # Random cropping:
    original_shape = tf.shape(stacked_images)[:2]
    num_image_channels, num_gt_channels = tf.shape(image)[-1], tf.shape(gt_image)[-1]
    random_scale_factor = tf.random_uniform([], minval=0.8, maxval=1., dtype=tf.float32)
    crop_shape = tf.to_int32(tf.to_float(original_shape) * random_scale_factor)
    crop_shape = tf.concat([crop_shape, [num_image_channels + num_gt_channels]], axis=0)
    stacked_images = tf.image.random_crop(stacked_images, crop_shape)

    # The remaining transformations should be applied either differently to the input and GT images
    # (nearest-neighbor resizing for the label image VS interpolated resizing for the image),
    # or only to the input image, not the GT one (color changes, etc.). Therefore, we split them back:
    image = stacked_images[..., :num_image_channels]
    gt_image = tf.cast(stacked_images[:, :, num_image_channels:], dtype=gt_image.dtype)

    # Resizing back to expected dimensions:
    image = tf.image.resize_images(image, original_shape)
    # unlike `resize_images()`, `resize_nearest_neighbor``only accept image batches as input.
    # So we artificially turn our single image into a batch before calling it:
    gt_image = tf.image.resize_nearest_neighbor(tf.expand_dims(gt_image, 0), original_shape)
    # then we squeeze it back into a simple image:
    gt_image = tf.squeeze(gt_image, 0)

    # Random B/S changes:
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.75)
    image = tf.clip_by_value(image, 0.0, 1.0)  # keeping pixel values in check

    return image, gt_image


def parse_function(filenames, resize_to=[226, 226], augment=True):
    """
    Parse files into input/label image pair.
    :param filenames:   Dict containing the two files (filenames['image'], filenames['label'])
    :param resize_to:   H x W Dimensions to resize the image and label to
    :param augment:     Flag to augment the pair
    :return:            Input tensor, Label tensor
    """

    img_filename, gt_filename = filenames['image'], filenames['label']

    # Reading the files and returning their content as bytes:
    image_string, gt_string = tf.read_file(img_filename), tf.read_file(gt_filename)
    # Decoding those into images
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    gt_decoded = tf.image.decode_png(gt_string, channels=1)

    # Converting image to float:
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    gt = tf.to_int32(gt_decoded)

    # Resizing:
    image = tf.image.resize_images(image, resize_to)
    # unlike `resize_images()`, `resize_nearest_neighbor``only accept image batches as input.
    # So we artificially turn our single image into a batch before calling it:
    gt = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(gt, 0), resize_to), 0)

    # Opt. augmenting the image:
    if augment:
        image, gt = _augmentation_fn(image, gt)

    return image, gt


def segmentation_input_fn(image_files, gt_files, resize_to=[256, 256],
                          shuffle=False, batch_size=32, num_epochs=None, augment=False,
                          seed=None):
    """
    Set up an input data pipeline for semantic segmentation applications.
    :param image_files:     List of input image files
    :param gt_files:        List of corresponding label image files
    :param resize_to:       H x W Dimensions to resize the image and label to
    :param shuffle:         Flag to shuffle the dataset
    :param batch_size:      Batch size
    :param num_epochs:      Number of epochs the dataset would be iterated over
    :param augment:         Flag to augment the pairs
    :param seed:            (opt) Seed
    :return:                tf.data.Dataset
    """
    # Converting to TF dataset:
    image_files = tf.constant(image_files)
    gt_files = tf.constant(gt_files)
    dataset = tf.data.Dataset.from_tensor_slices(
        {'image': image_files, 'label': gt_files})

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.prefetch(1)

    # Batching + adding parsing operation:
    parse_fn = functools.partial(parse_function, resize_to=resize_to, augment=augment)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(num_epochs)
    return dataset


def cityscapes_input_fn(split='train', root_folder=CITYSCAPES_FOLDER, resize_to=[256, 256],
                        shuffle=False, batch_size=32, num_epochs=None, augment=False,
                        seed=None):
    """
    Set up an input data pipeline for semantic segmentation applications on Cityscapes dataset.
    :param split:           Split name ('train', 'val', 'test')
    :param root_folder:     Cityscapes root folder
    :param resize_to:       H x W Dimensions to resize the image and label to
    :param shuffle:         Flag to shuffle the dataset
    :param batch_size:      Batch size
    :param num_epochs:      Number of epochs the dataset would be iterated over
    :param augment:         Flag to augment the pairs
    :param seed:            (opt) Seed
    :return:                tf.data.Dataset
    """

    input_files, gt_files = get_cityscapes_file_pairs(split=split, root_folder=root_folder)
    return segmentation_input_fn(input_files, gt_files,
                                 resize_to, shuffle, batch_size, num_epochs, augment, seed)


# -----------------------------------------------------------------------------
#  DISPLAY FUNCTIONS
# -----------------------------------------------------------------------------


def change_ratio(image=None, pred=None, gt=None, ratio=CITYSCAPES_IMG_RATIO):
    """
    Resze the images to the corresponding ratio.
    :param image:   (opt) Input image
    :param pred:    (opt) Predicted label image
    :param gt:      (opt) Target image
    :param ratio:   Ratio
    :return:        3 resized images
    """
    valid_input = image if image is not None else pred if pred is not None else gt
    current_size = tf.shape(valid_input)[-3:-1]
    width_with_ratio = tf.to_int32(tf.to_float(current_size[1]) * ratio)
    size_with_ratio = tf.stack([current_size[0], width_with_ratio], axis=0)
    if image is not None:
        image = tf.image.resize_images(image, size_with_ratio)
    if pred is not None:
        pred = tf.image.resize_nearest_neighbor(pred, size_with_ratio)
    if gt is not None:
        gt = tf.image.resize_nearest_neighbor(gt, size_with_ratio)
    return image, pred, gt


def convert_labels_to_colors_numpy(label, one_hot=True, num_classes=len(CITYSCAPES_LABELS),
                                   color_array=CITYSCAPES_COLORS, ignore_value=CITYSCAPES_IGNORE_VALUE):
    """
    Convert label images into color ones for display (for numpy objects).
    :param label:           Label image (numpy array)
    :param one_hot:         Flag if the label image hasn't been one-hot yet and therefore should
    :param num_classes:     Number of classes (for one-hotting)
    :param color_array:     Array mapping labels to colors
    :param ignore_value:    Value of label to be ignored (for one-hotting)
    :return:                Color map
    """

    if one_hot:
        label_shape = label.shape
        label = label.reshape(-1)
        label[label == ignore_value] = num_classes
        label = np.eye(num_classes + 1, dtype=np.int32)[label]
        label = label[..., :num_classes]
    else:
        label_shape = label.shape[:-1]
        label = label.reshape(-1, label.shape[-1])

    colors = np.matmul(label, color_array)

    return colors.reshape(list(label_shape) + [colors.shape[1]])


def convert_label_to_colors(label, one_hot=True, num_classes=len(CITYSCAPES_LABELS),
                            color_tensor=CITYSCAPES_COLORS_TF):
    """
    Convert label images into color ones for display (for Tensors).
    :param label:           Label image (Tensor)
    :param one_hot:         Flag if the label image hasn't been one-hot yet and therefore should
    :param num_classes:     Number of classes (for one-hotting)
    :param color_tensor:    Tensor mapping labels to colors
    :return:                Color map
    """

    label_shape = tf.shape(label)
    color_channels = tf.shape(color_tensor)[-1]

    if one_hot:
        label = tf.one_hot(label, num_classes)
    else:
        label_shape = label_shape[:-1]

    label = tf.reshape(tf.to_int32(label), (-1, num_classes))
    colors = tf.matmul(label, color_tensor)

    return tf.reshape(colors, tf.concat([label_shape, [color_channels]], axis=0))


def postprocess_to_show(image=None, pred=None, gt=None, one_hot=True, ratio=CITYSCAPES_IMG_RATIO):
    """
    Post-process the training results of a segmentation model (as Tensors), for display.
    :param image:       (opt.) Input image tensor
    :param pred:        (opt.) Predicted label map tensor
    :param gt:          (opt.) Target label map tensor
    :param one_hot:     Flag if the predicted label image hasn't been one-hot yet and therefore should
    :param ratio:       Original image ratio
    :return:            Processed image tensor(s)
    """
    out = []
    image_show, pred_show, gt_show = change_ratio(image, pred, gt,
                                                  ratio)
    if image is not None:
        out.append(image_show)

    if pred is not None:
        if one_hot:
            pred_show = tf.squeeze(pred_show, -1)  # removing unnecessary channel dimension
        pred_show = convert_label_to_colors(pred_show, one_hot=one_hot)
        out.append(pred_show)

    if gt is not None:
        gt_show = tf.squeeze(gt_show, -1)  # removing unnecessary channel dimension
        gt_show = convert_label_to_colors(gt_show)
        out.append(gt_show)

    return out if len(out) > 1 else out[0]