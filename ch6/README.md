# Chapter 6: Enhancing and Segmenting Images

In Chapter 6 of the book, several paradigms for _pixel-precise_ applications were covered. We introduced **_encoders-decoders_** and some specific architectures like **_U-Net_** and **_FCN_**. We presented how to apply them to multiple tasks from _**image denoising**_ to **_semantic segmentation_**. We also demonstrated how different solutions can be combined to tackle more advanced problems like instance **_segmentation_**.

In this folder, readers can find several notebooks tackling these various notions. Addtionally, some of the key code snippets are compiled into reusable Python files.

## Notebooks

- 6.1 - [Introduction to Auto-Encoders](./ch6_nb1_intro_to_autoencoders.ipynb)
    - Implement a simple fully-connected auto-encoder, and apply it to the MNIST dataset. Use the encoder to embed and visualize the full dataset, applying _t-SNE_.
- 6.2 - [Denoising with Auto-Encoders](./ch6_nb2_denoising_with_autoencoders.ipynb)
    - Reuse the previous auto-encoder to recover corrupted MNIST images.
- 6.3 - [Super-Resolution with Deep Auto-Encoders](./ch6_nb3_super_resolution_with_dae.ipynb)
    - Implement a basic convolutional deep auto-encoder, followed by a _U-Net_ model. Apply both to improving the quality of downscaled images.
- 6.4 - [Data Preparation for Smart Car Applications](./ch6_nb4_data_preparation_for_smart_car_apps.ipynb)
    - Discover the _Cityscapes_ dataset of urban scenes, and set up an efficient input pipeline to serve batches for semantic segmentation applications.
- 6.5 - [Semantic Segmentation for Smart Cars with FCN-8s](./ch6_nb5_semantic_segmentation_for_smart_cars_with_fcn8s.ipynb)
    - Implement a _FCN-8s_ model from scratch, and apply it to segmenting Cityscapes images for autonomous driving applications. Improve the results using per-class weighing of the loss.
- 6.6 - [Object and Instance Segmentation for Smart Cars with U-Net](./ch6_nb6_object_and_instance_segmentation_for_smart_cars_with_unet.ipynb)
    -  Apply our _U-Net_ model to segmenting Cityscapes, training it with the _Dice_ loss. Improve its results with _CRF_ post-processing and use a pre-train model to achieve instance segmentaton.
	
## Additional Files

- [cityscapes_utils.py](cityscapes_utils.py): functions to pre-process, serve, visualize the Cityscapes dataset (code presented in notebook [6.4](./ch6_nb4_data_preparation_for_smart_car_apps.ipynb)).
- [fcn.py](fcn.py): Keras implementation of the _FCN-8s_ model (code presented in notebook [6.5](./ch6_nb5_semantic_segmentation_for_smart_cars_with_fcn8s.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): Custom callbacks to monitor the training of Keras models (code presented in notebook [6.2](./ch6_nb2_denoising_with_autoencoders.ipynb)).
- [plot_utils.py](plot_utils.py): Plot helper functions (code presented in notebooks [6.1](./ch6_nb1_intro_to_autoencoders.ipynb)- [6.2](./ch6_nb2_denoising_with_autoencoders.ipynb)).
- [tf_losses_and_metrics.py](tf_losses_and_metrics.py): Custom TensorFlow/Keras losses and metrics for encoders-decoders (code presented in notebooks [6.1](./ch6_nb1_intro_to_autoencoders.ipynb)-[6.2](./ch6_nb2_denoising_with_autoencoders.ipynb)-[6.5](./ch6_nb5_semantic_segmentation_for_smart_cars_with_fcn8s.ipynb)-[6-6](./ch6_nb6_object_and_instance_segmentation_for_smart_cars_with_unet.ipynb)).
- [tf_math.py](tf_math.py): Advanced math/morphological functions in TensorFlow (code presented in notebooks [6.2](./ch6_nb2_denoising_with_autoencoders.ipynb)-[6.5](./ch6_nb5_semantic_segmentation_for_smart_cars_with_fcn8s.ipynb)).
- [unet.py](unet.py): Keras implementation of the _U-Net_ model (code presented in notebook [6.3](./ch6_nb3_super_resolution_with_dae.ipynb)).
