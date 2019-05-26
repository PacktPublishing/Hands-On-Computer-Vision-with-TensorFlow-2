**> Chapter 6:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# Enhancing and Segmenting Images

Convolutional neural networks can be built to output multi-dimensional data. Therefore, models can be trained to _predict images_. In Chapter 6, we introduced _encoders-decoders_ and the more specific _auto-encoders_, demonstrating how they can be applied to the recovery of corrupted images or to per-pixel classification (i.e. semantic segmentation). From simple digit images to pictures gathered for smart-car applications, the following notebooks explain how CNNs can edit and segment data.

## :notebook: Notebooks

(Reminder: Notebooks are better visualized with `nbviewer`: click [here](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch6) to continue on `nbviewer.jupyter.org`.)

- 6.1 - [Discovering Auto-Encoders](./ch6_nb1_discover_autoencoders.ipynb)
    - Build a simple _auto-encoder (AE)_ and explore its latent space (_data embedding_).
- 6.2 - [Denoising with Auto-Encoders](./ch6_nb2_denoise_with_autoencoders.ipynb)
    - Train the previous _auto-encoder_ to denoise corrupted images.
- 6.3 - [Improving Image Quality with Deep Auto-Encoders (Super-Resolution)](./ch6_nb3_improve_image_quality_with_dae.ipynb)
    - Implement a simple _convolutional auto-encoder_ followed by a more complex _U-Net_, and apply them to improving the resolution of low-quality images. 
- 6.4 - [Preparing Data for Smart Car Applications](./ch6_nb4_preparing_data_for_smart_car_apps.ipynb)
    - Discover and prepare _Cityscapes_, a famous dataset applied to the training of recognition systems for autonomous driving.
- 6.5 - [Building and Training a FCN-8s Model for Semantic Segmentation](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)
    - Extend the _VGG_ into _FCN-8s_, an efficient architecture for semantic segmentation; and apply it to visual recognition for autonomous driving. _Loss weighing_ strategies are also presented.
- 6.6 - [Building and Training a U-Net Model for Object and Instance Segmentation](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)
    - Apply the previously-implemented _U-Net_ architecture to visual recognition for autonomous driving, apply the _Dice_ loss, and reuse state-of-the-art algorithms to achieve _instance segmentation_.
	
## :page_facing_up: Additional Files

- [cityscapes_utils.py](cityscapes_utils.py): utility functions for the _Cityscapes_ dataset (code presented in notebook [6.4](./ch6_nb4_preparing_data_for_smart_car_apps.ipynb)).
- [fcn.py](fcn.py): functional implementation of the _FCN-8s_ architecture (code presented in notebook [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): custom Keras _callbacks_ to monitor the trainings of models (code presented in notebooks [4.1](../Chapter04/ch4_nb1_implement_resnet_from_scratch.ipynb) and [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [mnist_utils.py](mnist_utils.py): utility functions for the _MNIST_ dataset, using `tensorflow-datasets` (code presented in notebook [6.1](./ch6_nb1_discover_autoencoders.ipynb)).
- [plot_utils.py](plot_utils.py): utility functions to display results (code presented in notebook [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [tf_losses_and_metrics.py](tf_losses_and_metrics.py): custom losses and metrics to train/evalute CNNs (code presented in notebooks [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [tf_math.py](tf_math.py): custom mathematical functions reused in other scripts (code presented in notebooks [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [unet.py](unet.py): functional implementation of the _U-Net_ architecture  (code presented in notebook [6.3](./ch6_nb3_improve_image_quality_with_dae.ipynb)).
