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

import tensorflow as tf
from tf_math import log_n, binary_outline

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================

# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def initialize_variables():
    """
    Initialize uninitialized variables on the fly.
    Snippet by JihongJu (https://github.com/JihongJu/keras-fcn/blob/master/keras_fcn/metrics.py)
    """
    variables = tf.local_variables()
    uninitialized_variables = []
    for v in variables:
        if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
            uninitialized_variables.append(v)
            v._keras_initialized = True
    if uninitialized_variables:
        sess = tf.keras.backend.get_session()
        sess.run(tf.variables_initializer(uninitialized_variables))


def adapt_tf_streaming_metric_for_keras(tf_metric, name,
                                        preprocss_fn=None, postprocess_fn=None,
                                        **kwargs):
    """
    Wrap a TensorFlow metric into a partial function which can be passed to Keras models.
    :param tf_metric:      TensorFlow metric which should be wrapped (e.g. `tf.metrics.mean_iou`)
    :param name:           Should be the name of the TF metric function.
                           (e.g. for `tf.metrics.mean_iou`, `name` should be "mean_iou")
    :param preprocss_fn:   (opt.) Function to pre-process `y_true`, `y_pred`
    :param postprocess_fn: (opt.) Function to post-process the metric value
    :return:               Metric function compatible wth Keras
    """

    # Inspired from snipped by Ruzhitskiy Bogdan
    # (https://github.com/keras-team/keras/issues/6050#issuecomment-329996505)
    def metric(y_true, y_pred):
        """ Wrap a TF metric, instantiating hidden variables + updating hidden states"""

        # First we define the metric operation:
        value, update_op = tf_metric(y_true, y_pred, **kwargs)

        # It is possible that the metric is relying on some local variables,
        # that we should then initialize them:
        initialize_variables()

        # Finally, we force the update of the metric values:
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    def metric_with_pre_and_post_processing(y_true, y_pred):
        """ Wrapper to pre/post-process the data"""

        if preprocss_fn is not None:
            y_true, y_pred = preprocss_fn(y_true, y_pred)

        value = metric(y_true, y_pred)

        if postprocess_fn is not None:
            value = postprocess_fn(y_true, y_pred, value)

        return value

    metric_with_pre_and_post_processing.__name__ = name

    return metric_with_pre_and_post_processing


def get_mask_for_valid_labels(y_true, num_classes, ignore_value=255):
    """
    Build a mask for the valid pixels, i.e. those not belonging to the ignored classes.
    :param y_true:        Ground-truth label map(s) (each value represents a class trainID)
    :param num_classes:   Total nomber of classes
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none)
    :return:              Binary mask of same shape as `y_true`
    """
    mask_for_class_elements = y_true < num_classes
    mask_for_not_ignored = y_true != ignore_value
    mask = mask_for_class_elements & mask_for_not_ignored
    return mask


def prepare_data_for_segmentation_loss(y_true, y_pred, num_classes=10, ignore_value=255):
    """
    Prepare predicted logits and ground-truth maps for the loss, removing pixels from ignored classes.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param y_pred:        Predicted logit map(s) () (e.g., of shape B x H x W x N, N number of classes)
    :param num_classes:   Number of classes
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none) 
    :return:              Tensors edited, ready for the loss computation
    """

    with tf.name_scope('prepare_data_for_loss'):
        # Flattening the tensors to simplify the following operations:
        if len(y_pred.shape) > (len(y_true.shape) - 1):
            y_pred = tf.reshape(y_pred, [-1, num_classes])
        else:
            y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(tf.to_int32(y_true), [-1])

        if ignore_value is not None:
            # To compare only on the considered class, we remove all the elements in the images
            # belonging to the ignored ones.
            # For that, we first compute the mask of the pixels belonging to valid labels:
            mask_for_valid_labels = get_mask_for_valid_labels(
                y_true, num_classes, ignore_value=ignore_value)
    
            # Then we use this mask to remove all pixels/elements not belonging to valid classes:
            y_true = tf.boolean_mask(y_true, mask_for_valid_labels, axis=0, name='gt_valid')
            y_pred = tf.boolean_mask(y_pred, mask_for_valid_labels, axis=0, name='pred_valid')

    return y_true, y_pred


def prepare_class_weight_map(y_true, weights):
    """
    Prepare pixel weight map based on class weighing.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param weights:       1D tensor of shape (N,) containing the weight value for each of the N classes
    :return:              Weight map (e.g., of shape B x H x W)
    """
    y_true_one_hot = tf.one_hot(y_true, tf.shape(weights)[0])
    weight_map = tf.tensordot(y_true_one_hot, weights, axes=1)
    return weight_map


def prepare_outline_weight_map(y_true, num_classes, outline_size=5,
                               outline_val=4., default_val=1.):
    """
    Prepare pixel weight map based on class outlines.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param num_classes:   Number of classes
    :param outline_size:  Outline size/thickness
    :param outline_val:   Weight value for outline pixels
    :param default_val:   Weight value for other pixels
    :return:              Weight map (e.g., of shape B x H x W)
    """
    y_true_one_hot = tf.squeeze(tf.one_hot(y_true, num_classes), axis=-2)
    outline_map_perclass = binary_outline(y_true_one_hot, outline_size)
    outline_map = tf.reduce_max(outline_map_perclass, axis=-1)
    outline_map = outline_map * (outline_val - default_val) + default_val
    return outline_map


# -----------------------------------------------------------------------------
#  METRIC FUNCTIONS
# -----------------------------------------------------------------------------

def psnr(img_a, img_b, max_img_value=255):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two images.
    :param img_a:           Image A
    :param img_b:           Image B
    :param max_img_value:   Maximum possible pixel value of the images
    :return:                PSNR value
    """
    mse = tf.reduce_mean((img_a - img_b) ** 2)
    return 20 * log_n(max_img_value, 10) - 10 * log_n(mse, 10)


def segmentation_accuracy(y_true, y_pred, ignore_value=255):
    """
    Compute the accuracy, ignoring pixels from some misc. classes.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param y_pred:        Predicted logit map(s) () (e.g., of shape B x H x W x N, N number of classes)
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none) 
    :return:              Tensors edited, ready for the loss computation
    """
    with tf.name_scope('segmentation_accuracy'):
        num_classes = y_pred.shape[-1]
        y_pred = tf.to_int32(tf.argmax(y_pred, axis=-1))
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred, 
                                                            num_classes=num_classes, ignore_value=ignore_value)

        num_pixels_to_classify = tf.size(y_true)
        num_pixels_correct = tf.reduce_sum(tf.to_int32(tf.equal(y_true, y_pred, name='correct')))

        accuracy = tf.to_float(num_pixels_correct) / tf.to_float(num_pixels_to_classify)
        return accuracy
segmentation_accuracy.__name__ = "acc"


def mean_iou_metric(num_classes, ignore_value=255):
    """
    Return a IoU metric function for Keras models.
    :param num_classes:   Number of target classes
    :return:              Metric function compatible wth Keras
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none) 
    """

    def preprocess_for_mean_iou(y_true, y_pred):
        # Like for other metrics/losses, we remove the values for the ignored class(es):
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=num_classes, ignore_value=ignore_value)
        # And since tf.metrics.mean_iou() needs the label maps, not the one-hot versions,
        # we adapt accordingly:
        y_pred = tf.argmax(y_pred, axis=-1)
        # (y_true is already as a label map)

        return y_true, y_pred

    metric = adapt_tf_streaming_metric_for_keras(
        tf.metrics.mean_iou, "mean_iou", num_classes=num_classes,
        preprocss_fn=preprocess_for_mean_iou)

    return metric

# -----------------------------------------------------------------------------
#  LOSS FUNCTIONS
# -----------------------------------------------------------------------------

def segmentation_loss(y_true, y_pred, ignore_value=255):
    """
    Compute the cross-entropy loss for semantic segmentation, ignoring pixels from some misc. classes.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param y_pred:        Predicted logit map(s) () (e.g., of shape B x H x W x N, N number of classes)
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none) 
    :return:              Loss
    """
    with tf.name_scope('segmentation_loss'):
        num_classes = y_pred.shape[-1]
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=num_classes, ignore_value=ignore_value)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_pred,
                                                      reduction=tf.losses.Reduction.MEAN)
    return loss


def weighted_segmentation_loss(y_true, y_pred, weights, ignore_value=255):
    """
    Prepare predicted logits and ground-truth maps for the loss, removing pixels from ignored classes.
    :param y_true:        Ground-truth label map(s) (e.g., of shape B x H x W)
    :param y_pred:        Predicted logit map(s) () (e.g., of shape B x H x W x N, N number of classes)
    :param weights:       1D tensor of shape (N,) containing the weight value for each of the N classes
    :param ignore_value:  trainID value of ignored classes (`None` if ignored none) 
    :return:              Loss
    """
    with tf.name_scope('weighted_segmentation_loss'):
        num_classes = y_pred.shape[-1]
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true, y_pred,
                                                            num_classes=num_classes, ignore_value=ignore_value)
        weight_map = prepare_class_weight_map(y_true, weights)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_pred, weights=weight_map,
                                                      reduction=tf.losses.Reduction.MEAN)
    return loss