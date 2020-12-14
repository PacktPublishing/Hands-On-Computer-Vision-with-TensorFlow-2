## $5 Tech Unlocked 2021!
[Buy and download this Book for only $5 on PacktPub.com](https://www.packtpub.com/product/hands-on-computer-vision-with-tensorflow-2/9781788830645)
-----
*If you have read this book, please leave a review on [Amazon.com](https://www.amazon.com/gp/product/1788830644).     Potential readers can then use your unbiased opinion to help them make purchase decisions. Thank you. The $5 campaign         runs from __December 15th 2020__ to __January 13th 2021.__*

# Hands-On Computer Vision with TensorFlow 2

<a href="https://www.packtpub.com/application-development/hands-computer-vision-tensorflow-2?utm_source=github&utm_medium=repository&utm_campaign=9781788830645"><img src="./banner_images/book_cover.png" alt="Hands-On Computer Vision with TensorFlow 2" height="270px" align="right"></a>

**Leverage deep learning to create powerful image processing apps with TensorFlow 2.0 and Keras**

This is the code repository for [Hands-On Computer Vision with TensorFlow 2](https://www.packtpub.com/application-development/hands-computer-vision-tensorflow-2) by [Benjamin Planche](https://github.com/benjaminplanche) and [Eliot Andres](https://github.com/EliotAndres), published by Packt. 

This book is a practical guide to building high performance systems for object detection, segmentation, video processing, smartphone applications, and more. It is based on TensorFlow 2, the new version of Google's open-source library for machine learning.

This repository offers several notebooks to illustrate each of the chapters, as well as the complete sources for the advanced projects presented in the book. *Note that this repository is meant to complement the book. Therefore, we suggest to check out its content for more detailed explanations and advanced tips*.

## :mag_right: What is this book about?
Computer vision solutions are becoming increasingly common, making their way in fields such as health, automobile, social media, and robotics. This book will help you explore TensorFlow 2, the brand new version of Google's open source framework for machine learning. You will understand how to benefit from using convolutional neural networks (CNNs) for visual tasks. 

_Hands-On Computer Vision with TensorFlow 2_ starts with the fundamentals of computer vision and deep learning, teaching you how to build a neural network from scratch. You will discover the features that have made TensorFlow the most widely used AI library, along with its intuitive Keras interface, and move on to building, training, and deploying CNNs efficiently. Complete with concrete code examples, the book demonstrates how to classify images with modern solutions, such as Inception and ResNet, and extract specific content using You Only Look Once (YOLO), Mask R-CNN, and U-Net. You will also build Generative Adversarial Networks (GANs) and Variational Auto-Encoders (VAEs) to create and edit images, and LSTMs to analyze videos. In the process, you will acquire advanced insights into transfer learning, data augmentation, domain adaptation, and mobile and web deployment, among other key concepts. By the end of the book, you will have both the theoretical understanding and practical skills to solve advanced computer vision problems with TensorFlow 2.0.

This book covers the following exciting features:
* Create your own neural networks from scratch
* Classify images with modern architectures including Inception and ResNet
* Detect and segment objects in images with YOLO, Mask R-CNN, and U-Net
* Tackle problems in developing self-driving cars and facial emotion recognition systems
* Boost your application’s performance with transfer learning, GANs, and domain adaptation
* Use recurrent neural networks for video analysis
* Optimize and deploy your networks on mobile devices and in the browser

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1788830644) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## :wrench: Instructions and Navigation

If you’re new to deep learning and have some background in Python programming and image processing, like reading/writing image files and editing pixels, this book is for you. Even if you’re an expert curious about the new TensorFlow 2 features, you’ll find this book useful.
While some theoretical explanations require knowledge in algebra and calculus, the book covers concrete examples for learners focused on practical applications such as visual recognition for self-driving cars and smartphone apps.

The code is in the form of **[Jupyter](http://jupyter.org/) notebooks**. Unless specified otherwise, it is running using **Python 3.5 (or higher)** and **TensorFlow 2.0**. Installation instructions are presented in the book (we recommend [Anaconda](https://anaconda.org/) to manage the dependencies like [numpy](http://www.numpy.org/), [matplotlib](https://matplotlib.org), etc.).

As described in the following subsections, the provided Jupyter notebooks can either be studied directly or can be used as code recipes to run and reproduce the experiments presented in the book.

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/9781788830645_ColorImages.pdf).

### Study the Jupyter notebooks online

If you simply want to go through the provided code and results, you can directly access them online in the book's GitHub repository. Indeed, GitHub is able to render Jupyter notebooks and to display them as static web pages.
However, the GitHub viewer ignores some style formatting and interactive content. For the best online viewing experience, we recommend using instead **Jupyter nbviewer** (https://nbviewer.jupyter.org), an official web platform you can use to read Jupyter notebooks uploaded online. This website can be queried to render notebooks stored in GitHub repositories. Therefore, the Jupyter notebooks provided can also be read at the following address: https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2.

### Run the Jupyter notebooks on your machine

To read or run these documents on your machine, you should first install Jupyter Notebook. For those who already use Anaconda (https://www.anaconda.com) to manage and deploy their Python environments (as we will recommend in this book), Jupyter Notebook should be directly available (as it is installed with Anaconda). For those using other Python distributions and those not familiar with Jupyter Notebook, we recommend having a look at the documentation, which provides installation instructions and tutorials (https://jupyter.org/documentation).

Once Jupyter Notebook is installed on your machine, navigate to the directory containing the book's code files, open a terminal, and execute the following command:

    $ jupyter notebook
    
The web interface should open in your default browser. From there, you should be able to navigate the directory and open the Jupyter notebooks provided, either to read, execute, or edit them.

Some documents contain advanced experiments that can be extremely compute-intensive (such as the training of recognition algorithms over large datasets). Without the proper acceleration hardware (that is, without compatible NVIDIA GPUs, as explained in Chapter 2, _TensorFlow Basics and Training a Model_), these scripts can take hours or even days (even with compatible GPUs, the most advanced examples can take quite some time).

### Run the Jupyter notebooks in Google Colab

For those who wish to run the Jupyter notebooks themselves—or play with new experiments—but do not have access to a powerful enough machine, we recommend using **Google Colab**, also named Colaboratory (https://colab.research.google.com). It is a cloud-based Jupyter environment, provided by Google, for people to run compute-intensive scripts on powerful machines.

### Software and Hardware List 
With the following software and hardware list you can run all code files present in the book (Chapter 1-9).

| Chapter| Software required                                           | OS required                        |
| -------| ------------------------------------------------------------| ---------------------------------- |
| 1-9    | Jupyter Notebook                                            | Windows, Mac OS X, and Linux (Any) |
| 1-9    | Python 3.5 and above, NumPy, Matplotlib, Anaconda (Optional)| Windows, Mac OS X, and Linux (Any) |
| 2-9    | TensorFlow, tensorflow-gpu                                  | Windows, Mac OS X, and Linux (Any) |
| 3      | Scikit-Image                                                | Windows, Mac OS X, and Linux (Any) |
| 4      | TensorFlow Hub                                              | Windows, Mac OS X, and Linux (Any) |
| 6      | pydensecrf library                                          | Windows, Mac OS X, and Linux (Any) |
| 7      | Vispy, Plyfile                                              | Windows, Mac OS X, and Linux (Any) |
| 8      | opencv-python, tqdm, scikit-learn                           | Windows, Mac OS X, and Linux (Any) |
| 9      | Android Studio, Cocoa Pods, Yarn                            | Windows, Mac OS X, and Linux (Any) |


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
 - Chapter 5 - [Object Detection Models](/Chapter05)
    - 5.1 - [Running YOLO inference](./Chapter05/ch5_nb1_yolo_inference.ipynb)
    - 5.2 - (TBD) Training a YOLO model
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
    - 7.6 - [Applying DANN to Train the Segmentation Model on Synthetic Data](./Chapter07/ch7_nb6_apply_dann_to_train_segmentation_model_on_synthetic_data.ipynb)
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

## :busts_in_silhouette: Get to Know the Authors
**Benjamin Planche**
is a passionate PhD student at the University of Passau and Siemens Corporate Technology. He has been working in various research labs around the world (LIRIS in France, Mitsubishi Electric in Japan, and Siemens in Germany) in the fields of computer vision and deep learning for more than five years. Benjamin has a double master's degree with first-class honors from INSA-Lyon, France, and the University of Passau, Germany.
His research efforts are focused on developing smarter visual systems with less data, targeting industrial applications. Benjamin also shares his knowledge and experience on online platforms, such as StackOverflow, or applies this knowledge to the creation of aesthetic demos.


**Eliot Andres**
is a freelance deep learning and computer vision engineer. He has more than 3 years' experience in the field, applying his skills to a variety of industries, such as banking, health, social media, and video streaming. Eliot has a double master's degree from École des Ponts and Télécom, Paris.
His focus is industrialization: delivering value by applying new technologies to business problems. Eliot keeps his knowledge up to date by publishing articles on his blog and by building prototypes using the latest technologies.

## :scroll: Referencing

If you use the code samples in your study/work or want to cite the book, please use:

```bibtex
@book{Andres_Planche_HandsOnCVWithTF2,
 author = {Planche, Benjamin and Andres, Eliot},
 title = {Hands-On Computer Vision with TensorFlow 2},
 year = {2019},
 isbn = {978-1788830645},
 publisher = {Packt Publishing Ltd},
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

## Errata
* Page 18: **stand of the heart** _should be_ **state-of-the-art**
* Page 24: **graphical processing unit** _should be_ **graphics processing unit**
* Page 55: **before hand** _should be_ **beforehand**
* Page 76: **indiacting** _should be_ **indicating**
* Page 90: **depth dimensions into a single vector** _should be_ **depth dimensions into a single dimension**
* Page 178: **bceause** _should be_ **because**
* Page 183: **smaller than the input and target latent spaces** _should be_ **smaller than the input and target spaces**
* Page 214: **cannot only** _should be_ **can not only**
* Page 254: **Jupyter Notebooks** _should be_ **Jupyter notebooks**

### Related products
* Mastering OpenCV 4 with Python [[Packt]](https://www.packtpub.com/application-development/mastering-opencv-4-python?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1789344913)

* OpenCV 4 for Secret Agents - Second Edition [[Packt]](https://www.packtpub.com/application-development/opencv-4-secret-agents-second-edition?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1789345367)


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.


