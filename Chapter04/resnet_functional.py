"""
File name: resnet_functional.py
Author: Benjamin Planche
Date created: 26.03.2019
Date last modified: 18:56 26.03.2019
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, 
    GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, add)
import tensorflow.keras.regularizers as regulizers

#==============================================================================
# Function Definitions
#==============================================================================

def _res_conv(filters, kernel_size=3, padding='same', strides=1, use_relu=True, use_bias=False, name='cbr',
              kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
    """
    Return a layer block chaining conv, batchnrom and reLU activation.
    :param filters:                 Number of filters.
    :param kernel_size:             Kernel size.
    :param padding:                 Convolution padding.
    :param strides:                 Convolution strides.
    :param use_relu:                Flag to apply ReLu activation at the end.
    :param use_bias:                Flag to use bias or not in Conv layer.
    :param name:                    Name suffix for the layers.
    :param kernel_initializer:      Kernel initialisation method name.
    :param kernel_regularizer:      Kernel regularizer.
    :return:                        Callable layer block
    """

    def layer_fn(x):
        conv = Conv2D(
            filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, 
            name=name + '_c')(x)
        res = BatchNormalization(axis=-1, name=name + '_bn')(conv)
        if use_relu:
            res = Activation("relu", name=name + '_r')(res)
        return res

    return layer_fn


def _merge_with_shortcut(kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4), 
                         name='block'):
    """
    Return a layer block which merge an input tensor and the corresponding 
    residual output tensor from another branch.
    :param kernel_initializer:      Kernel initialisation method name.
    :param kernel_regularizer:      Kernel regularizer.
    :param name:                    Name suffix for the layers.
    :return:                        Callable layer block
    """

    def layer_fn(x, x_residual):
        # We check if `x_residual` was scaled down. If so, we scale `x` accordingly with a 1x1 conv:
        x_shape = tf.keras.backend.int_shape(x)
        x_residual_shape = tf.keras.backend.int_shape(x_residual)
        if x_shape == x_residual_shape:
            shortcut = x
        else:
            strides = (
                int(round(x_shape[1] / x_residual_shape[1])), # vertical stride
                int(round(x_shape[2] / x_residual_shape[2]))  # horizontal stride
            )
            x_residual_channels = x_residual_shape[3]
            shortcut = Conv2D(
                filters=x_residual_channels, kernel_size=(1, 1), padding="valid", strides=strides,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_shortcut_c')(x)

        merge = add([shortcut, x_residual])
        return merge

    return layer_fn


def _residual_block_basic(filters, kernel_size=3, strides=1, use_bias=False, name='res_basic',
                          kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
    """
    Return a basic residual layer block.
    :param filters:                 Number of filters.
    :param kernel_size:             Kernel size.
    :param strides:                 Convolution strides
    :param use_bias:                Flag to use bias or not in Conv layer.
    :param kernel_initializer:      Kernel initialisation method name.
    :param kernel_regularizer:      Kernel regularizer.
    :return:                        Callable layer block
    """

    def layer_fn(x):
        x_conv1 = _res_conv(
            filters=filters, kernel_size=kernel_size, padding='same', strides=strides, 
            use_relu=True, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr_1')(x)
        x_residual = _res_conv(
            filters=filters, kernel_size=kernel_size, padding='same', strides=1, 
            use_relu=False, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr_2')(x_conv1)
        merge = _merge_with_shortcut(kernel_initializer, kernel_regularizer,name=name)(x, x_residual)
        merge = Activation('relu')(merge)
        return merge

    return layer_fn


def _residual_block_bottleneck(filters, kernel_size=3, strides=1, use_bias=False, name='res_bottleneck',
                               kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
    """
    Return a residual layer block with bottleneck, recommended for deep ResNets (depth > 34).
    :param filters:                 Number of filters.
    :param kernel_size:             Kernel size.
    :param strides:                 Convolution strides
    :param use_bias:                Flag to use bias or not in Conv layer.
    :param kernel_initializer:      Kernel initialisation method name.
    :param kernel_regularizer:      Kernel regularizer.
    :return:                        Callable layer block
    """

    def layer_fn(x):
        x_bottleneck = _res_conv(
            filters=filters, kernel_size=1, padding='valid', strides=strides, 
            use_relu=True, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr1')(x)
        x_conv = _res_conv(
            filters=filters, kernel_size=kernel_size, padding='same', strides=1, 
            use_relu=True, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr2')(x_bottleneck)
        x_residual = _res_conv(
            filters=filters * 4, kernel_size=1, padding='valid', strides=1, 
            use_relu=False, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr3')(x_conv)
        merge = _merge_with_shortcut(kernel_initializer, kernel_regularizer, name=name)(x, x_residual)
        merge = Activation('relu')(merge)
        return merge

    return layer_fn


def _residual_macroblock(block_fn, filters, repetitions=3, kernel_size=3, strides_1st_block=1, use_bias=False,
                         kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4),
                         name='res_macroblock'):
    """
    Return a layer block, composed of a repetition of `N` residual blocks.
    :param block_fn:                Block layer method to be used.
    :param repetitions:             Number of times the block should be repeated inside.
    :param filters:                 Number of filters.
    :param kernel_size:             Kernel size.
    :param strides_1st_block:       Convolution strides for the 1st block.
    :param use_bias:                Flag to use bias or not in Conv layer.
    :param kernel_initializer:      Kernel initialisation method name.
    :param kernel_regularizer:      Kernel regularizer.
    :return:                        Callable layer block
    """

    def layer_fn(x):
        for i in range(repetitions):
            block_name = "{}_{}".format(name, i) 
            strides = strides_1st_block if i == 0 else 1
            x = block_fn(filters=filters, kernel_size=kernel_size, 
                         strides=strides, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                         name=block_name)(x)
        return x

    return layer_fn


def ResNet(input_shape, num_classes=1000, block_fn=_residual_block_basic, repetitions=(2, 2, 2, 2),
           use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
    """
    Build a ResNet model for classification.
    :param input_shape:             Input shape (e.g. (224, 224, 3))
    :param num_classes:             Number of classes to predict
    :param block_fn:                Block layer method to be used.
    :param repetitions:             List of repetitions for each macro-blocks the network should contain.
    :param use_bias:                Flag to use bias or not in Conv layer.
    :param kernel_initializer:      Kernel initialisation method name.
    :param kernel_regularizer:      Kernel regularizer.
    :return:                        ResNet model.
    """

    # Input and 1st layers:
    inputs = Input(shape=input_shape)
    conv = _res_conv(
        filters=64, kernel_size=7, strides=2, use_relu=True, use_bias=use_bias,
        kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inputs)
    maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv)

    # Chain of residual blocks:
    filters = 64
    strides = 2
    res_block = maxpool
    for i, repet in enumerate(repetitions):
        # We do not further reduce the input size for the 1st block (max-pool applied just before):
        block_strides = strides if i != 0 else 1
        macroblock_name = "block_{}".format(i) 
        res_block = _residual_macroblock(
            block_fn=block_fn, repetitions=repet, name=macroblock_name,
            filters=filters, strides_1st_block=block_strides, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(res_block)
        filters = min(filters * 2, 1024) # we limit to 1024 filters max

    # Final layers for prediction:
    res_spatial_dim = tf.keras.backend.int_shape(res_block)[1:3]
    avg_pool = AveragePooling2D(pool_size=res_spatial_dim, strides=1)(res_block)
    flatten = Flatten()(avg_pool)
    predictions = Dense(units=num_classes, kernel_initializer=kernel_initializer, 
                        activation='softmax')(flatten)

    # Model:
    model = Model(inputs=inputs, outputs=predictions)
    return model


def ResNet18(input_shape, num_classes=1000, use_bias=True,
             kernel_initializer='he_normal', kernel_regularizer=None):
    return ResNet(input_shape, num_classes, block_fn=_residual_block_basic, repetitions=(2, 2, 2, 2),
                  use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


def ResNet34(input_shape, num_classes=1000, use_bias=True,
             kernel_initializer='he_normal', kernel_regularizer=None):
    return ResNet(input_shape, num_classes, block_fn=_residual_block_basic, repetitions=(3, 4, 6, 3),
                  use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


def ResNet50(input_shape, num_classes=1000, use_bias=True,
             kernel_initializer='he_normal', kernel_regularizer=None):
    # Note: ResNet50 is similar to ResNet34,
    # with the basic blocks replaced by bottleneck ones (3 conv layers each instead of 2)
    return ResNet(input_shape, num_classes, block_fn=_residual_block_bottleneck, repetitions=(3, 4, 6, 3),
                  use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


def ResNet101(input_shape, num_classes=1000, use_bias=True,
             kernel_initializer='he_normal', kernel_regularizer=None):
    return ResNet(input_shape, num_classes, block_fn=_residual_block_bottleneck, repetitions=(3, 4, 23, 3),
                  use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


def ResNet152(input_shape, num_classes=1000, use_bias=True,
             kernel_initializer='he_normal', kernel_regularizer=None):
    return ResNet(input_shape, num_classes, block_fn=_residual_block_bottleneck, repetitions=(3, 8, 36, 3),
                  use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)