"""
File name: plot_utils.py
Author: Benjamin Planche
Date created: 14.02.2019
Date last modified: 14:42 14.02.2019
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

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#==============================================================================
# Function Definitions
#==============================================================================

def plot_image_grid(images, titles=None, figure=None,
                    grayscale=False, transpose=False):
    """
    Plot a grid of n x m images
    :param images:       Images in a n x m array
    :param titles:       (opt.) List of m titles for each image column
    :param figure:       (opt.) Pyplot figure (if None, will be created)
    :param grayscale:    (opt.) Flag to draw the images in grayscale
    :param transpose:    (opt.) Flag to transpose the grid
    :return:             Pyplot figure filled with the images
    """
    num_cols, num_rows = len(images), len(images[0])
    img_ratio = images[0][0].shape[1] / images[0][0].shape[0]

    if transpose:
        vert_grid_shape, hori_grid_shape = (1, num_rows), (num_cols, 1)
        figsize = (int(num_rows * 5 * img_ratio), num_cols * 5)
        wspace, hspace = 0.2, 0.
    else:
        vert_grid_shape, hori_grid_shape = (num_rows, 1), (1, num_cols)
        figsize = (int(num_cols * 5 * img_ratio), num_rows * 5)
        hspace, wspace = 0.2, 0.

    if figure is None:
        figure = plt.figure(figsize=figsize)
    imshow_params = {'cmap': plt.get_cmap('gray')} if grayscale else {}
    grid_spec = gridspec.GridSpec(*hori_grid_shape, wspace=0, hspace=0)

    for j in range(num_cols):
        grid_spec_j = gridspec.GridSpecFromSubplotSpec(
            *vert_grid_shape, subplot_spec=grid_spec[j], wspace=wspace, hspace=hspace)

        for i in range(num_rows):
            ax_img = figure.add_subplot(grid_spec_j[i])
            # ax_img.axis('off')
            ax_img.set_yticks([])
            ax_img.set_xticks([])
            if titles is not None:
                if transpose:
                    ax_img.set_ylabel(titles[j], fontsize=25)
                else:
                    ax_img.set_title(titles[j], fontsize=15)
            ax_img.imshow(images[j][i], **imshow_params)

    figure.tight_layout()
    return figure


def figure_to_rgb_array(fig):
    """
    Convert figure into a RGB array
    :param fig:         PyPlot Figure
    :return:            RGB array
    """
    figure_buffer = io.BytesIO()
    fig.savefig(figure_buffer, format='png')
    figure_buffer.seek(0)
    figure_string = figure_buffer.getvalue()
    return figure_string


def figure_to_summary(fig, name, step):
    """
    Convert figure into TF summary
    :param fig:             Figure
    :param tag:             Summary name
    :return:                Summary step
    """
    # Transform figure into PNG buffer:
    figure_string = figure_to_rgb_array(fig)

    # Transform PNG buffer into image tensor:
    figure_tensor = tf.image.decode_png(figure_string, channels=4)
    figure_tensor = tf.expand_dims(figure_tensor, 0) # adding batch dimension

    # Using Proto to convert the image string into a summary:
    figure_summary = tf.summary.image(name, figure_tensor, step)

    # # Using Proto to convert the image string into a summary:
    # figure_summary_image = tf.Summary.Image(
    #     # height=input_images.shape[1], width=input_images.shape[2],
    #     # colorspace=4,
    #     encoded_image_string=figure_string)
    # figure_summary = tf.Summary(
    #     value=[tf.Summary.Value(tag=tag, image=figure_summary_image)])

    return figure_summary
