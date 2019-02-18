"""
File name: tf_losses_and_metrics.py
Author: Benjamin Planche
Date created: 14.02.2019
Date last modified: 14:58 14.02.2019
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
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Lambda, Dropout, MaxPooling2D, LeakyReLU, concatenate, BatchNormalization)

#==============================================================================
# Function Definitions
#==============================================================================


def name_layer_factory(num=0, name_prefix="", name_suffix=""):
    """
    Helper function to name all our layers.
    """

    def name_layer_fn(layer):
        return '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    return name_layer_fn


def conv_bn_lrelu(filters, kernel_size=3, batch_norm=True,
                  kernel_initializer='he_normal', padding='same',
                  name_fn=lambda layer: "conv_bn_lrelu-{}".format(layer)):
    """
    Return a function behaving like a sequence convolution + BN + lReLU.
    :param filters:              Number of filters for the convolution
    :param kernel_size:          Kernel size for the convolutions
    :param batch_norm:           Flag to perform batch normalization
    :param kernel_initializer:   Name of kernel initialization method
    :param padding:              Name of padding option
    :param name_fn:              Function to name each layer of this sequence
    :return:                     Function chaining layers
    """

    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size,
                   activation=None, kernel_initializer=kernel_initializer,
                   padding=padding, name=name_fn('conv'))(x)
        if batch_norm:
            x = BatchNormalization(name=name_fn('bn'))(x)
        x = LeakyReLU(alpha=0.3, name=name_fn('act'))(x)
        return x

    return block


def unet_conv_block(filters, kernel_size=3,
                    batch_norm=True, dropout=False,
                    name_prefix="enc_", name_suffix=0):
    """
    Return a function behaving like a U-Net convolution block.
    :param filters:              Number of filters for the convolution
    :param kernel_size:          Kernel size for the convolutions
    :param batch_norm:           Flag to perform batch normalization
    :param dropout:              NFlag to perform dropout between the two convs
    :param name_prefix:          Prefix for the layer names
    :param name_suffix:          FSuffix for the layer names
    :return:                     Function chaining layers
    """

    def block(x):
        # First convolution:
        name_fn = name_layer_factory(1, name_prefix, name_suffix)
        x = conv_bn_lrelu(filters, kernel_size=kernel_size, batch_norm=batch_norm,
                          name_fn=name_layer_factory(1, name_prefix, name_suffix))(x)
        if dropout:
            x = Dropout(0.2, name=name_fn('drop'))(x)

        # Second convolution:
        name_fn = name_layer_factory(2, name_prefix, name_suffix)
        x = conv_bn_lrelu(filters, kernel_size=kernel_size, batch_norm=batch_norm,
                          name_fn=name_layer_factory(2, name_prefix, name_suffix))(x)

        return x

    return block


# As the dimensions of our images are not normalized, and often not even, it is
# possible that after downsampling and upsampling, we do not reobtain the original size
# (with a difference of +/- 1px).
# To avoid the problems this may cause, we define a layer to slightly resize the generated
# image to the dimensions of the target one:
ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize_images(images[0], tf.shape(images[1])[1:3]),
    # `images` is a tuple of 2 tensors.
    # We resize the first image tensor to the shape of the 2nd
    name=name)


def unet(x, out_channels=3, layer_depth=4, filters_orig=32, kernel_size=4,
         batch_norm=True, dropout=True, final_activation='sigmoid'):
    """
    Define a U-Net network.
    :param x:                    Input tensor/placeholder
    :param out_channels:         Number of output channels
    :param filters_orig:         Number of filters for the 1st CNN layer
    :param kernel_size:          Kernel size for the convolutions
    :param batch_norm:           Flag to perform batch normalization
    :param dropout:              Flag to perform dropout
    :param final_activation:     Name of activation function for the final layer
    :return:                     Network (Keras Functional API)
    """

    # Encoding layers:
    filters = filters_orig
    outputs_for_skip = []
    for i in range(layer_depth):
        # Convolution block:
        x_conv = unet_conv_block(filters, kernel_size,
                                 dropout=dropout, batch_norm=batch_norm,
                                 name_prefix="enc_", name_suffix=i)(x)

        # We save the pointer to the output of this encoding block,
        # to pass it to its parallel decoding block afterwards:
        outputs_for_skip.append(x_conv)

        # Downsampling:
        x = MaxPooling2D(2)(x_conv)

        filters = min(filters * 2, 512)

    # Bottleneck layers:
    x = unet_conv_block(filters, kernel_size, dropout=dropout,
                        batch_norm=batch_norm, name_suffix='_btleneck')(x)

    # Decoding layers:
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)

        # Upsampling:
        name_fn = name_layer_factory(3, "ups_", i)
        x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=2,
                            activation=None, kernel_initializer='he_normal',
                            padding='same', name=name_fn('convT'))(x)
        if batch_norm:
            x = BatchNormalization(name=name_fn('bn'))(x)
        x = LeakyReLU(alpha=0.3, name=name_fn('act'))(x)

        # Concatenation with the output of the corresponding encoding block:
        shortcut = outputs_for_skip[-(i + 1)]
        x = ResizeToSame(name='resize_to_same{}'.format(i))([x, shortcut])

        x = concatenate([x, shortcut], axis=-1, name='dec_conc{}'.format(i))

        # Convolution block:
        use_dropout = dropout and (i < (layer_depth - 2))
        x = unet_conv_block(filters, kernel_size,
                            batch_norm=batch_norm, dropout=use_dropout,
                            name_prefix="dec_", name_suffix=i)(x)

    x = Conv2D(filters=out_channels, kernel_size=1, activation=final_activation,
               padding='same', name='dec_output')(x)

    return x