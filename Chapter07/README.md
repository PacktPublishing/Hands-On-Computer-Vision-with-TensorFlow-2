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
- 7.3 - (TBD) Rendering Images from 3D Models
    - Get a quick overview of 3D rendering with Python, using _OpenGL_-based `vispy` to generate a variety of images from 3D data. 
- 7.4 - (TBD) Apply Domain Adaptation Methods to Bridge the Realism Gap
    - Experiment with some solutions, such as _DANN_, to train models on synthetic data so that they can be applied to real pictures afterwards.
- 7.5 - (TBD) Create Images with Variational Auto-Encoders (VAEs)
    - Implement a particular _auto-encoder_ able to generate new real-looking images.
- 7.6 - (TBD) Create Images with Generative-Adversarial Networks (GANs)
    - Train a _generative_ network against a _discriminator_ one in an unsupervised manner, to augment datasets.
	
## :page_facing_up: Additional Files

- [cityscapes_utils.py](cityscapes_utils.py): utility functions for the _Cityscapes_ dataset (code presented in notebook [6.4](./ch6_nb4_preparing_data_for_smart_car_apps.ipynb)).
- [fcn.py](fcn.py): functional implementation of the _FCN-8s_ architecture (code presented in notebook [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): custom Keras _callbacks_ to monitor the trainings of models (code presented in notebooks [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb) and [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [mnist_utils.py](mnist_utils.py): utility functions for the _MNIST_ dataset, using `tensorflow-datasets` (code presented in notebook [6.1](./ch6_nb1_discover_autoencoders.ipynb)).
- [plot_utils.py](plot_utils.py): utility functions to display results (code presented in notebook [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [tf_losses_and_metrics.py](tf_losses_and_metrics.py): custom losses and metrics to train/evalute CNNs (code presented in notebooks [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [tf_math.py](tf_math.py): custom mathematical functions reused in other scripts (code presented in notebooks [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [unet.py](unet.py): functional implementation of the _U-Net_ architecture  (code presented in notebook [6.3](./ch6_nb3_improve_image_quality_with_dae.ipynb)).
