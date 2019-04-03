"""
File name: mnist_utils.py
Author: Benjamin Planche
Date created: 28.03.2019
Date last modified: 16:26 28.03.2019
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

import tensorflow as tf
import tensorflow_datasets as tfds
import functools

#==============================================================================
# Constant Definitions
#==============================================================================

MNIST_BUILDER = tfds.builder("mnist")
MNIST_BUILDER.download_and_prepare()

#==============================================================================
# Function Definitions
#==============================================================================

def get_info():
    """
    Return the Tensorflow-Dataset info for MNIST.
    :return:            Dataset info
    """
    return MNIST_BUILDER.info


def _prepare_data_fn(features, target='label', flatten=True,
                     return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimensions, and opt. apply some random transformations.
    :param features:              Data
    :param target                 Target/ground-truth data to be returned along the images
                                  ('label' for categorical labels, 'image' for images, or None)
    :param flatten:               Flag to flatten the images, from (28, 28, 1) to (784,)
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """
    
    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.
    
    image = features['image']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    if flatten:
        is_batched = len(image.shape) > 3
        flattened_shape = (-1, 784) if is_batched else (784,)
        image = tf.reshape(image, flattened_shape)
        
    if target is None:
        return image if return_batch_as_tuple else {'image': image}
    else:
        features['image'] = image
        return (image, features[target]) if return_batch_as_tuple else features

    
def get_mnist_dataset(phase='train', target='label', batch_size=32, num_epochs=None, 
                      shuffle=True, flatten=True, return_batch_as_tuple=True, seed=None):
    """
    Instantiate a CIFAR-100 dataset.
    :param phase:                 Phase ('train' or 'val')
    :param target                 Target/ground-truth data to be returned along the images
                                  ('label' for categorical labels, 'image' for images, or None)
    :param batch_size:            Batch size
    :param num_epochs:            Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle:               Flag to shuffle the dataset (if True)
    :param flatten:               Flag to flatten the images, from (28, 28, 1) to (784,)
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Iterable Dataset
    """

    assert(phase == 'train' or phase == 'test')
    
    prepare_data_fn = functools.partial(_prepare_data_fn, return_batch_as_tuple=return_batch_as_tuple,
                                        target=target, flatten=flatten, seed=seed)

    mnist_dataset = MNIST_BUILDER.as_dataset(split=tfds.Split.TRAIN if phase =='train' else tfds.Split.TEST)
    mnist_dataset = mnist_dataset.repeat(num_epochs)
    if shuffle:
        mnist_dataset = mnist_dataset.shuffle(10000, seed=seed)
    mnist_dataset = mnist_dataset.batch(batch_size)
    mnist_dataset = mnist_dataset.map(prepare_data_fn, num_parallel_calls=4)
    mnist_dataset = mnist_dataset.prefetch(1)
    
    return mnist_dataset