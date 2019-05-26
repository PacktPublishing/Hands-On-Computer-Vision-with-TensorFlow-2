**> Chapter 7:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# Training on Complex and Scarce Datasets

The first task to develop new recognition models is to gather and prepare the training dataset. Building pipelines to let the data flow properly during heavy training phases used to be an art, but TensorFlow recent features made it quite straightforward to fetch and pre-process complex data, as demonstrated in the first notebooks of this Chapter 7. Oftentimes, however, training data can simply be unavailable. The remaining notebooks tackle these scenarios, presenting a variety of solutions.

## :notebook: Notebooks

(Reminder: Notebooks are better visualized with `nbviewer`: click [here](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch7) to continue on `nbviewer.jupyter.org`.)

- 7.1 - [Setting up Efficient Input Pipelines with `tf.data`](./ch7_nb1_set_up_efficient_input_pipelines_with_tf_data.ipynb)
    - Harness the latest features of the `tf.data` API to set up optimized input pipelines to train models.
- 7.2 - [Generating and Parsing TFRecords](./ch7_nb2_generate_and_parse_tfrecords.ipynb)
    - Discover how to convert complete datasets into _TFRecords_, and how to efficiently parse these files.
- 7.3 - [Rendering Images from 3D Models](./ch7_nb3_render_images_from_3d_models.ipynb)
    - Get a quick overview of 3D rendering with Python, using _OpenGL_-based `vispy` to generate a variety of images from 3D data. 
- 7.4 - [Training a Segmentation Model on Synthetic Images](./ch7_nb4_train_segmentation_model_on_synthetic_images.ipynb)
    - Use a pre-rendered dataset of synthetic images to train a model, and evaluate the effects of the *realism gap* on its final accuracy. 
- 7.5 - [Training a Simple Domain Adversarial Network](./ch7_nb5_train_a_simple_domain_adversarial_network_(dann).ipynb)
    - Discover and implement a famous domain adaptation method: *DANN*. 
- 7.6 - [Applying DANN to Train the Segmentation Model on SYnthetic Data](./ch7_nb6_apply_dann_to_train_segmentation_model_on_synthetic_data.ipynb)
    - Apply the previous DANN method to improve the performance of our segmentation model suffering from the *realism gap*. 
- 7.7 - [Generating Images with VAEs](./ch7_nb7_generate_images_with_vae_models.ipynb)
    - Build your first generative neural network, a simple *Variational Auto-Encoder (VAE), to create digit images. 
- 7.8 - [Generating Images with GANs](./ch7_nb8_generate_images_with_gan_models.ipynb)
    - Discover another famous type of generative neural models: the *Generative Adversarial Networks (GANs). 
	
## :page_facing_up: Additional Files

- [cityscapes_utils.py](cityscapes_utils.py): utility functions for the _Cityscapes_ dataset (code presented in notebook [6.4](../Chapter06/ch6_nb4_preparing_data_for_smart_car_apps.ipynb)).
- [fcn.py](fcn.py): functional implementation of the _FCN-8s_ architecture (code presented in notebook [6.5](../Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): custom Keras _callbacks_ to monitor the trainings of models (code presented in notebooks [4.1](../Chapter04/ch4_nb1_implement_resnet_from_scratch.ipynb) and [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [plot_utils.py](plot_utils.py): utility functions to display results (code presented in notebook [6.2](../Chapter06/ch6_nb2_denoise_with_autoencoders.ipynb)).
- [renderer.py](renderer.py): object-oriented pipeline to render images from 3D models (code presented in notebook [7.3](./ch7_nb3_render_images_from_3d_models.ipynb)).
- [synthia_utils.py](synthia_utils.py): utility functions for the _SYNTHIA_ dataset (code presented in notebook [7.4](./ch7_nb4_train_segmentation_model_on_synthetic_images.ipynb)).
- [tf_losses_and_metrics.py](tf_losses_and_metrics.py): custom losses and metrics to train/evalute CNNs (code presented in notebooks [6.5](../Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](../Chapter06/ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [tf_math.py](tf_math.py): custom mathematical functions reused in other scripts (code presented in notebooks [6.5](../Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](../Chapter06/ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
