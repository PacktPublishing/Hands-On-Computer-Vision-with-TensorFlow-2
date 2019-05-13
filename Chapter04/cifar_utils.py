"""
File name: cifar_utils.py
Author: Benjamin Planche
Date created: 26.03.2019
Date last modified: 18:55 26.03.2019
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

import tensorflow as tf
import tensorflow_datasets as tfds
import functools

#==============================================================================
# Constant Definitions
#==============================================================================

CIFAR_BUILDER = tfds.builder("cifar100")
CIFAR_BUILDER.download_and_prepare()

#==============================================================================
# Function Definitions
#==============================================================================


def _prepare_data_fn(features, input_shape, augment=False, return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimensions, and opt. apply some random transformations.
    :param features:              Data
    :param input_shape:           Shape expected by the models (images will be resized accordingly)
    :param augment:               Flag to apply some random augmentations to the images
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """
    input_shape = tf.convert_to_tensor(input_shape)
    
    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.
    
    image = features['image']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    if augment:
        # Randomly applied horizontal flip:
        image = tf.image.random_flip_left_right(image, seed=seed)

        # Random B/S changes:
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0) # keeping pixel values in check

        # Random resize and random crop back to expected size:
        
        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4, dtype=tf.float32, seed=seed)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale_factor, 
                                tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale_factor, 
                               tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape, seed=seed)
    else:
        image = tf.image.resize(image, input_shape[:2])
        
    if return_batch_as_tuple:
        label = features['label']
        features = (image, label)
    else:
        features['image'] = image
    return features


def get_info():
    """
    Return the Tensorflow-Dataset info for CIFAR-100.
    :return:            Dataset info
    """
    return CIFAR_BUILDER.info


def get_dataset(phase='train', batch_size=32, num_epochs=None, shuffle=True,
                input_shape=(32, 32, 3), return_batch_as_tuple=True, seed=None):
    """
    Instantiate a CIFAR-100 dataset.
    :param phase:                 Phase ('train' or 'val')
    :param batch_size:            Batch size
    :param num_epochs:            Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle:               Flag to shuffle the dataset (if True)
    :param input_shape:           Shape of the processed images
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Iterable Dataset
    """

    assert(phase == 'train' or phase == 'test')
    is_train = phase == 'train'
    
    prepare_data_fn = functools.partial(_prepare_data_fn, return_batch_as_tuple=return_batch_as_tuple,
                                        input_shape=input_shape, augment=is_train, seed=seed)

    cifar_dataset = CIFAR_BUILDER.as_dataset(split=tfds.Split.TRAIN if phase =='train' else tfds.Split.TEST)
    cifar_dataset = cifar_dataset.repeat(num_epochs)
    if shuffle:
        cifar_dataset = cifar_dataset.shuffle(10000, seed=seed)
    
    cifar_dataset = cifar_dataset.map(prepare_data_fn, num_parallel_calls=4)
    cifar_dataset = cifar_dataset.batch(batch_size)
    cifar_dataset = cifar_dataset.prefetch(1)
    
    return cifar_dataset