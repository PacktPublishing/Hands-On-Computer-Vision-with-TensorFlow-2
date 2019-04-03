# The ImageNet dataset is provided by the ImageNet team, Princeton University, and Stanford University.
# The Tiny-ImageNet is provided by Stanford University,
# for the course project "CS231n: Convolutional Neural Networks for Visual Recognition" (http://cs231n.stanford.edu/).

# Tiny-ImageNet can be downloaded from https://tiny-imagenet.herokuapp.com/ or http://image-net.org/download-images (users need the proper access).

"""
File name: tiny_imagenet_utils.py
Author: Benjamin Planche
Date created: 21.03.2019
Date last modified: 17:50 21.03.2019
Python Version: "3.6"

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

import os
import glob
from functools import partial
import tensorflow as tf

#==============================================================================
# Constant Definitions
#==============================================================================

ROOT_FOLDER = os.path.expanduser('~/datasets/tiny-imagenet-200/')
IMAGENET_IDS_FILE_BASENAME = 'wnids.txt'
IMAGENET_WORDS_FILE_BASENAME = 'words.txt'
NUM_CLASSES = 200
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64, 64, 3


#==============================================================================
# Function Definitions
#==============================================================================

def _get_class_information(ids_file, words_file):
    """
    Extract the class IDs and corresponding human-readable labels from metadata files.
    :param ids_file:        IDs filename (contains list of unique string class IDs)
    :param words_file:      Words filename (contains list of tuples <ID, human-readable label>)
    :return:                List of IDs, Dictionary of labels
    """
    with open(ids_file, "r") as f:
        class_ids = [line[:-1] for line in f.readlines()] # removing the `\n` for each line

    with open(words_file, "r") as f:
        words_lines = f.readlines()
        class_readable_labels = {}
        for line in words_lines:
            # We split the line between the ID (9-char long) and the human readable label:
            class_id = line[:9]
            class_label = line[10:-1]

            # If this class is in our dataset, we add it to our id-to-label dictionary:
            if class_id in class_ids:
                class_readable_labels[class_id] = class_label

    return class_ids, class_readable_labels


def _get_train_image_files_and_labels(root_folder, class_ids):
    """
    Fetch the lists of training images and numerical labels.
    We assume the images are stored as "<root_folder>/train/<class_id>/images/*.JPEG"
    :param root_folder:     Dataset root folder
    :param class_ids:       List of class IDs
    :return:                List of image filenames and List of corresponding labels
    """
    image_files, labels = [], []

    for i in range(len(class_ids)):
        class_id = class_ids[i]
        # Grabbing all the image files for this class:
        class_image_paths = os.path.join(root_folder, 'train', class_id, 'images', '*.JPEG')
        class_images = glob.glob(class_image_paths)
        # Creating as many numerical labels:
        class_labels = [i] * len(class_images)

        image_files += class_images
        labels += class_labels

    return image_files, labels


def _get_val_image_files_and_labels(root_folder, class_ids):
    """
    Fetch the lists of validation images and numerical labels.
    We assume the images are stored as "<root_folder>/train/<class_id>/images/*.JPEG"
    :param root_folder:     Dataset root folder
    :param class_ids:       List of class IDs
    :return:                List of image filenames and List of corresponding labels
    """
    image_files, labels = [], []
    val_images_folder = os.path.join(root_folder, 'val', 'images')

    # The file 'val_annotations.txt' contains for each line the image filename and its annotations.
    # We parse it to build our dataset lists:
    val_annotation_file = os.path.join(root_folder, 'val', 'val_annotations.txt')
    with open(val_annotation_file, "r") as f:
        anno_lines = f.readlines()
        for line in anno_lines:
            split_line = line.split('\t')   # Splitting the line to extract the various pieces of info
            if len(split_line) > 1:
                image_file, image_class_id = split_line[0], split_line[1]
                class_num_id = class_ids.index(image_class_id)
                if class_num_id >= 0:   # If the label belongs to our dataset, we add them:
                    image_files.append(os.path.join(val_images_folder, image_file))
                    labels.append(class_num_id)

    return image_files, labels


def _parse_function(filename, label, size=[IMG_HEIGHT, IMG_WIDTH]):
    """
    Parse the provided tensors, loading and resizing the corresponding image.
    Code snippet from https://www.tensorflow.org/guide/datasets#decoding_image_data_and_resizing_it (Apache 2.0 License).
    :param filename:    Image filename (String Tensor)
    :param label:       Image label
    :return:            Image, Label
    """
    # Reading the file and returning its content as bytes:
    image_string = tf.io.read_file(filename)
    # Decoding those into the image
    # (with `channels=3`, TF will duplicate the channels of grayscale images so they have 3 channels too):
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    # Converting to float:
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # Resizing the image to the expected dimensions:
    image_resized = tf.image.resize(image_float, size)
    return image_resized, label


def _input_fn(image_files, image_labels,
              shuffle=True, batch_size=32, num_epochs=None,
              augmentation_fn=None, wrap_for_estimator=True, resize_to=None):
    """
    Prepares and returns the iterators for a dataset.
    :param image_files:         List of image files
    :param image_labels:        List of image labels
    :param shuffle:             Flag to shuffle the dataset (if True)
    :param batch_size:          Batch size
    :param num_epochs:          Number of epochs (to repeat the iteration - infinite if None)
    :param augmentation_fn:     opt. Augmentation function
    :param wrap_for_estimator:  Flag to wrap the inputs to be passed for Estimators
    :param resize_to:           (opt) Dimensions (h x w) to resize the images to
    :return:                    Iterable batched images and labels
    """

    # Converting to TF dataset:
    image_files = tf.constant(image_files)
    image_labels = tf.constant(image_labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_files, image_labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50000)
    # Adding parsing operation, to open and decode images:
    if resize_to is None:
        parse_fn = _parse_function
    else:
        # We specify to which dimensions to resize the images, if requested:
        parse_fn = partial(_parse_function, size=resize_to)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    # Opt. adding some further transformations:
    if augmentation_fn is not None:
        dataset.map(augmentation_fn, num_parallel_calls=4)
    # Further preparing for iterating on:
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(1)
    if wrap_for_estimator:
        dataset = dataset.map(lambda img, label: ({'image': img}, label))
    return dataset


def _training_augmentation_fn(image, label):
    """
    Apply random transformations to augment the training images.
    :param images:      Images
    :param label:       Labels
    :return:            Augmented Images, Labels
    """

    # Randomly applied horizontal flip:
    image = tf.image.random_flip_left_right(image)

    # Random B/S changes:
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0) # keeping pixel values in check

    # Random resize and random crop back to expected size:
    original_shape = tf.shape(image)
    random_scale_factor = tf.random.uniform([1], minval=0.7, maxval=1.3, dtype=tf.float32)
    scaled_height = tf.cast(tf.cast(original_shape[0], tf.float32) * random_scale_factor, tf.int32)
    scaled_width = tf.cast(tf.cast(original_shape[1], tf.float32) * random_scale_factor,  tf.int32)
    scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
    image = tf.image.resize(image, scaled_shape)
    image = tf.image.random_crop(image, original_shape)

    return image, label


def tiny_imagenet(phase='train', shuffle=True, batch_size=32, num_epochs=None,
                  augmentation_fn=_training_augmentation_fn, wrap_for_estimator=True,
                  root_folder=ROOT_FOLDER, resize_to=None):
    """
    Instantiate a Tiny-Image training or validation dataset, which can be passed to any model.
    :param phase:               Phase ('train' or 'val')
    :param shuffle:             Flag to shuffle the dataset (if True)
    :param batch_size:          Batch size
    :param num_epochs:          Number of epochs (to repeat the iteration - infinite if None)
    :param augmentation_fn:     opt. Augmentation function
    :param wrap_for_estimator:  Flag to wrap the inputs to be passed for Estimators
    :param root_folder:         Dataset root folder
    :param resize_to:           (opt) Dimensions (h x w) to resize the images to
    :return:                    Dataset, IDs List, Dictionary to read labels
    """

    ids_file = os.path.join(root_folder, IMAGENET_IDS_FILE_BASENAME)
    words_file = os.path.join(root_folder, IMAGENET_WORDS_FILE_BASENAME)
    class_ids, class_readable_labels = _get_class_information(ids_file, words_file)
    if phase == 'train':
        image_files, image_labels = _get_train_image_files_and_labels(root_folder, class_ids)
    elif phase == 'val':
        image_files, image_labels = _get_val_image_files_and_labels(root_folder, class_ids)
    else:
        raise ValueError("Unknown phase ('train' or 'val' only)")

    dataset = _input_fn(image_files, image_labels,
                               shuffle, batch_size, num_epochs, augmentation_fn,
                               wrap_for_estimator, resize_to)

    return dataset, class_ids, class_readable_labels


def tiny_imagenet_input_fn(phase='train', shuffle=True, batch_size=32, num_epochs=None,
                           augmentation_fn=_training_augmentation_fn,
                           root_folder=ROOT_FOLDER, resize_to=None):
    """
    Instantiate a Tiny-Image training or validation dataset, which can be passed to an Estimator.
    :param phase:               Phase ('train' or 'val')
    :param shuffle:             Flag to shuffle the dataset (if True)
    :param batch_size:          Batch size
    :param num_epochs:          Number of epochs (to repeat the iteration - infinite if None)
    :param augmentation_fn:     opt. Augmentation function
    :param wrap_for_estimator:  Flag to wrap the inputs to be passed for Estimators
    :param root_folder:         Dataset root folder
    :return:                    Iterator for the inputs, Iterator for the labels, IDs List, Dictionary to read labels
    """

    ids_file = os.path.join(root_folder, IMAGENET_IDS_FILE_BASENAME)
    words_file = os.path.join(root_folder, IMAGENET_WORDS_FILE_BASENAME)
    class_ids, class_readable_labels = _get_class_information(ids_file, words_file)
    if phase == 'train':
        image_files, image_labels = _get_train_image_files_and_labels(root_folder, class_ids)
    elif phase == 'val':
        image_files, image_labels = _get_val_image_files_and_labels(root_folder, class_ids)
    else:
        raise ValueError("Unknown phase ('train' or 'val' only)")

    input_fn = lambda: _input_fn(
        image_files, image_labels, shuffle, batch_size, num_epochs, augmentation_fn, True, resize_to)

    return input_fn
