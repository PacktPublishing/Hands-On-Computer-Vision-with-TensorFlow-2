"""
File name: tf_math
Author: Benjamin Planche
Date created: 17.02.2019
Date last modified: 17:12 17.02.2019
Python Version: "3.6"

Copyright = "Copyright (C) 2018-2019 of Packt"
Credits = ["Eliot Andres, Benjamin Planche"] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
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

#==============================================================================
# Function Definitions
#==============================================================================

def log_n(x, n=10):
    """
    Compute log_n(x), i.e. the log base `n` value of `x`.
    :param x:   Input tensor
    :param n:   Value of the log base
    :return:    Log result
    """
    log_e = tf.math.log(x)
    div_log_n = tf.math.log(tf.constant(n, dtype=log_e.dtype))
    return log_e / div_log_n


def binary_dilation(x, kernel_size=3):
    """
    Apply dilation of the given binary tensor (each input channel is processed independently)
    :param x:               Binary tensor of shape BxHxWxC
    :param kernel_size:     Kernel size  (int or Tensor)
    :return:                Dilated Tensor
    """
    with tf.name_scope("binary_dilation"):
        num_channels = tf.shape(x)[-1]
        kernel = tf.ones((kernel_size, kernel_size, num_channels, 1), dtype=x.dtype)
        conv = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        clip = tf.clip_by_value(conv, 1., 2.) - 1.
        return clip


def binary_erosion(x, kernel_size=3):
    """
    Apply erosion of the given binary tensor (each input channel is processed independently)
    :param x:              Binary tensor of shape BxHxWxC or HxWxC (if is_single_image)
    :param kernel_size:    Kernel size  (int or Tensor)
    :return:               Eroded Tensor
    """
    with tf.name_scope("binary_erosion"):
        num_channels = tf.shape(x)[-1]
        kernel = tf.ones((kernel_size, kernel_size, num_channels, 1), dtype=x.dtype)
        conv = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        max_val = tf.constant(kernel_size * kernel_size, dtype=x.dtype)
        clip = tf.clip_by_value(conv, max_val - 1, max_val)
        return clip - (max_val - 1)


def binary_opening(tensor, kernel_size=3):
    """
    Apply opening of the given binary tensor (each input channel is processed independently)
    :param tensor:                  Binary tensor of shape BxHxWxC or HxWxC (if is_single_image)
    :param kernel_size:             Kernel size  (int or Tensor)
    :return:                        Tensor
    """
    with tf.name_scope("binary_opening"):
        return binary_dilation(binary_erosion(tensor, kernel_size), kernel_size)


def binary_closing(x, kernel_size=3):
    """
    Apply closing of the given binary tensor (each input channel is processed independently)
    :param tensor:                  Binary tensor of shape BxHxWxC or HxWxC (if is_single_image)
    :param kernel_size:             Kernel size  (int or Tensor)
    :return:                        Tensor
    """
    with tf.name_scope("binary_opening"):
        return binary_erosion(binary_dilation(x, kernel_size), kernel_size)


def binary_outline(x, kernel_size=3):
    """
    Compute the outline (cf. cv2.morphologyEx) of the given binary tensor
    (each input channel is processed independently)
    :param x:                  Binary tensor of shape BxHxWxC or HxWxC (if is_single_image)
    :param kernel_size:        Kernel size  (int or Tensor)
    :return:                   Tensor
    """
    with tf.name_scope("binary_outline"):
        return binary_dilation(x, kernel_size) - binary_erosion(x, kernel_size)