"""
File name: keras_custom_callbacks.py
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

import collections
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from plot_utils import plot_image_grid, figure_to_summary

#==============================================================================
# Class Definitions
#==============================================================================

class DynamicPlotCallback(tf.keras.callbacks.Callback):
    """ Keras callback which plots the training losses/metrics and updates the figure after each epoch:.
    """

    def on_train_begin(self, logs={}):
        # This method will be called when the training start.
        # Therefore, we use it to initialize some elements for our Callback:
        self.logs = dict()
        self.fig, self.ax = None, None

    def on_epoch_end(self, epoch, logs={}):
        # This method will be called after each epoch.
        # Keras will call this function, providing the current epoch number,
        # and the values of the various losses/metrics for this epoch (`logs` dict).

        # We add the new log values to the list...
        for key, val in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(val)
        # ... then we plot everything:
        self._plot_logs()

    def on_train_end(self, logs={}):
        pass  # our callback does nothing special at the end of the training

    def on_epoch_begin(self, epoch, logs={}):
        pass  # ... nor at the beginning of a new epoch

    def on_batch_begin(self, batch, logs={}):
        pass  # ... nor at the beginning of a new batch

    def on_batch_end(self, batch, logs={}):
        pass  # ... nor after.

    def _plot_logs(self):
        # Method to clear the figures and draw them over with new values:
        if self.fig is None:  # First call - we initialize the figure:
            num_metrics = len(self.logs)
            self.fig, self.ax = plt.subplots(math.ceil(num_metrics / 2), 2, figsize=(10, 8))
            self.fig.show()
            self.fig.canvas.draw()

        # Plotting:
        i = 0
        for key, val in self.logs.items():
            id_vert, id_hori = i // 2, i % 2
            self.ax[id_vert, id_hori].clear()
            self.ax[id_vert, id_hori].set_title(key)
            self.ax[id_vert, id_hori].plot(val)
            i += 1

        # self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.75, bottom=0.25)
        self.fig.canvas.draw()

#==============================================================================

class SimpleLogCallback(tf.keras.callbacks.Callback):
    """ Keras callback for simple, denser console logs."""

    def __init__(self, metrics_dict, num_epochs='?', log_frequency=1,
                 metric_string_template='\033[1m[[name]]\033[0m = \033[94m{[[value]]:5.3f}\033[0m'):
        """
        Initialize the Callback.
        :param metrics_dict:            Dictionary containing mappings for metrics names/keys
                                        e.g. {"accuracy": "acc", "val. accuracy": "val_acc"}
        :param num_epochs:              Number of training epochs
        :param log_frequency:           Log frequency (in epochs)
        :param metric_string_template:  (opt.) String template to print each metric
        """
        super().__init__()

        self.metrics_dict = collections.OrderedDict(metrics_dict)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency

        # We build a format string to later print the metrics, (e.g. "Epoch 0/9: loss = 1.00; val-loss = 2.00")
        log_string_template = 'Epoch {0:2}/{1}: '
        separator = '; '

        i = 2
        for metric_name in self.metrics_dict:
            templ = metric_string_template.replace('[[name]]', metric_name).replace('[[value]]', str(i))
            log_string_template += templ + separator
            i += 1

        # We remove the "; " after the last element:
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

    def on_train_begin(self, logs=None):
        print("Training: \033[92mstart\033[0m.")

    def on_train_end(self, logs=None):
        print("Training: \033[91mend\033[0m.")

    def on_epoch_end(self, epoch, logs={}):
        if (epoch - 1) % self.log_frequency == 0 or epoch == self.num_epochs:
            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]
            print(self.log_string_template.format(epoch, self.num_epochs, *values))

#==============================================================================

class TensorBoardImageGridCallback(tf.keras.callbacks.Callback):
    """ Keras callback for generative models, to draw grids of
        input/predicted/target images into Tensorboard every epoch.
    """

    def __init__(self, log_dir, input_images, target_images=None, tag='images',
                 figsize=(10, 10), dpi=300, grayscale=False, transpose=False,
                 preprocess_fn=None):
        """
        Initialize the Callback.
        :param log_dir:         Folder to write the image summaries into
        :param input_images:    List of input images to use for the grid
        :param target_images:   (opt.) List of target images for the grid
        :param tag:             Tag to name the Tensorboard summary
        :param figsize:         Pyplot figure size for the grid
        :param dpi:             Pyplot figure DPI
        :param grayscale:       Flag to plot the images as grayscale
        :param transpose:       Flag to transpose the image grid
        :param preprocess_fn:   (opt.) Function to pre-process the
                                input/predicted/target image lists before plotting
        """
        super().__init__()

        self.summary_writer = tf.summary.create_file_writer(log_dir)

        self.input_images, self.target_images = input_images, target_images
        self.tag = tag
        self.postprocess_fn = preprocess_fn

        self.image_titles = ['images', 'predicted']
        if self.target_images is not None:
            self.image_titles.append('ground-truth')

        # Initializing the figure:
        self.fig = plt.figure(num=0, figsize=figsize, dpi=dpi)
        self.grayscale = grayscale
        self.transpose = transpose

    def on_epoch_end(self, epoch, logs={}):
        """
        Plot into Tensorboard a grid of image results.
        :param epoch:   Epoch num
        :param logs:    (unused) Dictionary of loss/metrics value for the epoch
        """

        # Get predictions with current model:
        predicted_images = self.model.predict_on_batch(self.input_images)
        if self.postprocess_fn is not None:
            input_images, predicted_images, target_images = self.postprocess_fn(
                self.input_images, predicted_images, self.target_images)
        else:
            input_images, target_images = self.input_images, self.target_images

        # Fill figure with images:
        grid_imgs = [input_images, predicted_images]
        if target_images is not None:
            grid_imgs.append(target_images)
        self.fig.clf()
        self.fig = plot_image_grid(grid_imgs, titles=self.image_titles, figure=self.fig,
                                   grayscale=self.grayscale, transpose=self.transpose)

        with self.summary_writer.as_default():
            # Transform into summary:
            figure_summary = figure_to_summary(self.fig, self.tag, epoch)

            # # Finally, log it:
            # self.summary_writer.add_summary(figure_summary, global_step=epoch)
        self.summary_writer.flush()

    def on_train_end(self, logs={}):
        """
        Close the resources used to plot the grids.
        :param logs:    (unused) Dictionary of loss/metrics value for the epoch
        """
        self.summary_writer.close()
        plt.close(self.fig)