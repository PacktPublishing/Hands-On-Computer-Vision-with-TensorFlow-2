# Hands On Computer Vision with TensorFlow 2
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="./banner_images/book_cover.png" width=200 align="right">
</a>

_Leverage deep learning to create powerful computer vision apps with TensorFlow 2.0 and Keras._

This repository contains the code for the book ***Hands On Computer Vision with TensorFlow 2*** by [Benjamin Planche](https://github.com/Aldream) and [Eliot Andres](https://github.com/EliotAndres), published by [Packt](https://www.packtpub.com/?utm_source=github).

More precisely, this repository offers several notebooks to illustrate each of the chapters and their notions, as well as the complete sources for the advanced projects used as examples along the book. Note that this repository is meant to complement the book. Therefore, we suggest to check out its content for more detailed explanations and advanced tips.

## :mag_right: About the Book

Computer vision is achieving a new frontier of capabilities in artificial intelligence including medical screening, self-driving cars and expression detection. TensorFlow is one of the most widely used AI frameworks that leverages deep convolutional neural networks to process complex data. This book explores Google's open source TensorFlow 2 framework along its Keras API, and teaches how to apply them to solving advanced computer vision tasks. It will help you acquire the skills and understand vital concepts to be a part of the extraordinary advances in this domain. 

## :wrench: Technical Requirements

The code is in the form of [Jupyter](http://jupyter.org/) notebooks. Unless specified otherwise, it is running using Python 3.5 (or higher) and TensorFlow 2.0. Installation instructions are presented in the book (we recommend [Anaconda](https://anaconda.org/) to manage the dependencies like [numpy](http://www.numpy.org/), [matplotlib](https://matplotlib.org), etc.).

## :books: Table of Content

- Chapter 1 - [Computer Vision and Neural Networks](/Chapter01)
    - 1.1 - [Building and Training a Neural Network from Scratch](./Chapter01/ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)
- Chapter 2 - [TensorFlow Basics and Training a Model ](/Chapter02)
    - 2.1 - [Training a model with Keras](./Chapter02/ch2_nb1_mnist_keras.ipynb)
- Chapter 3 - [Modern Neural Networks](/Chapter03)
    - 3.1 - [Discovering CNNs' Basic Operations](./Chapter03/ch3_nb1_discover_cnns_basic_ops.ipynb)
    - 3.2 - [Building and Training our First CNN with TensorFlow 2 and Keras](./Chapter03/ch3_nb2_build_and_train_first_cnn_with_tf2.ipynb)
    - 3.3 - [Experimenting with Advanced Optimizers](./Chapter03/ch3_nb3_experiment_with_optimizers.ipynb)
    - 3.4 - [Applying Regularization Methods to CNNs](./Chapter03/ch3_nb4_apply_regularization_methods_to_cnns.ipynb)
- Chapter 4 - [Influential Classification Tools](/Chapter04)
    - 4.1 - [Implementing ResNet from Scratch](./Chapter04/ch4_nb1_implement_resnet_from_scratch.ipynb)
    - 4.2 - [Reusing Models from Keras Applications](./Chapter04/ch4_nb2_reuse_models_from_keras_apps.ipynb)
    - 4.3 - [Fetching Models from TensorFlow Hub](./Chapter04/ch4_nb3_fetch_models_from_tf_hub.ipynb)
    - 4.4 - [Applying Transfer Learning](./Chapter04/ch4_nb4_apply_transfer_learning.ipynb)
    - 4.5 - (Appendix) [Exploring ImageNet and Tiny-ImageNet](./Chapter04/ch4_nb5_explore_imagenet_and_its_tiny_version.ipynb)
 - Chapter 5
    - 5.1 - (TBD) Training a YOLO model
 - Chapter 6 - [Enhancing and Segmenting Images](./Chapter06)
    - 6.1 - [Discovering Auto-Encoders](./Chapter06/ch6_nb1_discover_autoencoders.ipynb)
    - 6.2 - [Denoising with Auto-Encoders](./Chapter06/ch6_nb2_denoise_with_autoencoders.ipynb)
    - 6.3 - [Improving Image Quality with Deep Auto-Encoders (Super-Resolution)](./Chapter06/ch6_nb3_improve_image_quality_with_dae.ipynb)
    - 6.4 - [Preparing Data for Smart Car Applications](./Chapter06/ch6_nb4_preparing_data_for_smart_car_apps.ipynb)
    - 6.5 - [Building and Training a FCN-8s Model for Semantic Segmentation](./Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)
    - 6.6 - [Building and Training a U-Net Model for Object and Instance Segmentation](./Chapter06/ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)
    - 6.6 - [Object and Instance Segmentation for Smart Cars with U-Net](./Chapter06/ch6_nb6_object_and_instance_segmentation_for_smart_cars_with_unet.ipynb)
- Chapter 7 - [Training on Complex and Scarce Datasets](/Chapter07)
    - 7.1 - [Setting up Efficient Input Pipelines with `tf.data`](./Chapter07/ch7_nb1_set_up_efficient_input_pipelines_with_tf_data.ipynb)
    - 7.2 - [Generating and Parsing TFRecords](./Chapter07/ch7_nb2_generate_and_parse_tfrecords.ipynb)
    - 7.3 - [Rendering Images from 3D Models](./Chapter07/ch7_nb3_render_images_from_3d_models.ipynb)
    - 7.4 - [Training a Segmentation Model on Synthetic Images](./Chapter07/ch7_nb4_train_segmentation_model_on_synthetic_images.ipynb)
    - 7.5 - [Training a Simple Domain Adversarial Network](./Chapter07/ch7_nb5_train_a_simple_domain_adversarial_network_(dann).ipynb)
    - 7.6 - [Applying DANN to Train the Segmentation Model on SYnthetic Data](./Chapter07/ch7_nb6_apply_dann_to_train_segmentation_model_on_synthetic_data.ipynb)
    - 7.7 - [Generating Images with VAEs](./Chapter07/ch7_nb7_generate_images_with_vae_models.ipynb)
    - 7.8 - [Generating Images with GANs](./Chapter07/ch7_nb8_generate_images_with_gan_models.ipynb) 	
- Chapter 8 - [Video and Recurrent Neural Networks ](/Chapter08)
    - 8.1 - [Action recognition using an LSTM](./Chapter08/ch8_nb1_action_recognition.ipynb)
- Chapter 9 - [Optimizing Models and Deploying on Mobile Devices](/Chapter09)
    - 9.1 - [Model profiling](./Chapter09/ch9_nb1_profiling.ipynb)
    - 9.2 - [Non-maximum suppression algorithm comparison](./Chapter09/ch9_nb2_nms_speed_comparison.ipynb)
    - 9.3 - [Training an emotion detection model and converting it for mobile devices](./Chapter09/ch9_nb3_train_model.ipynb)
    - [iOS app](./Chapter09/coreml_ios)
    - [Android app](./Chapter09/tf_lite_android)
    - [Tensorflow.js app](./Chapter09/tfjs)

## :scroll: Citing

If you use the code samples in your study/work or want to cite the book, please use:

```bibtex
@book{Andres_Planche_HandsOnCVWithTF2,
 author = {Planche, Benjamin and Andres, Eliot},
 title = {Hands-On Computer Vision with TensorFlow 2},
 year = {2019},
 isbn = {TBD},
 publisher = {Packt Publishing},
}
```

<details><summary>Other Formats: (Click to View)</summary>
    <br/>
    <table>
        <tbody>
            <tr>
                <th scope="row">MLA</th>
                <td>Planche, Benjamin and Andres, Eliot. <i>Hands-On Computer Vision with TensorFlow 2</i>. Packt Publishing Ltd, 2019.</td>
            </tr>
            <tr>
                <th scope="row">APA</th>
                <td>Planche B., & Andres, E. (2019). <i>Hands-On Computer Vision with TensorFlow 2</i>. Packt Publishing Ltd.</td>
            </tr>
            <tr>
                <th scope="row">Chicago</th>
                <td>Planche, Benjamin, and Andres, Eliot. <i>Hands-On Computer Vision with TensorFlow 2</i>. Packt Publishing Ltd, 2019.</td>
            </tr>
            <tr>
                <th scope="row">Harvard</th>
                <td>Planche B. and Andres, E., 2019. <i>Hands-On Computer Vision with TensorFlow 2</i>. Packt Publishing Ltd.</td>
            </tr>
            <tr>
                <th scope="row">Vancouver</th>
                <td>Planche B, Andres E. Hands-On Computer Vision with TensorFlow 2. Packt Publishing Ltd; 2019.</td>
            </tr>
        </tbody>
    </table>
<p>
    <a href="https://scholar.googleusercontent.com">EndNote</a> 
    <a href="https://scholar.googleusercontent.com">RefMan</a>
    <a href="https://scholar.googleusercontent.com" target="RefWorksMain">RefWorks</a>
</p>
</details>
