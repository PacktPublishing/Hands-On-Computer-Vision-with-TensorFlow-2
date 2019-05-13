**> Chapter 4:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# Influential Classification Tools

Chapter 4 is dedicated to influential CNN architectures like _VGG_, _Inception_, _ResNet_, etc., detailing their contributions to computer vision and machine learning in general. Introducing more complex classification tasks, we also explain how CNNs can benefit from knowledge acquired on different datasets (_transfer learning_). Therefore, the following notebooks contain the detailed implementation of some of these influential models, and present how to efficiently reuse pre-implemented and pre-trained models shared on various platforms. 

## :notebook: Notebooks

(Reminder: Notebooks are better visualized with `nbviewer`: click [here](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch4) to continue on `nbviewer.jupyter.org`.)

- 4.1 - [Implementing ResNet from Scratch](./ch4_nb1_implement_resnet_from_scratch.ipynb)
    - Implement block by block the very-deep _ResNet_ architecture (_ResNet-18_, _ResNet-50_, _ResNet-152_) and apply to the classification of a large dataset (_CIFAR-100_) obtained through `tensorflow-datasets`.
- 4.2 - [Reusing Models from Keras Applications](./ch4_nb2_reuse_models_from_keras_apps.ipynb)
    - Discover how to reuse pre-implemented models available in `keras.applications`, training another version of _ResNet-50_.
- 4.3 - [Fetching Models from TensorFlow Hub](./ch4_nb3_fetch_models_from_tf_hub.ipynb)
    - Navigate [tfhub.dev](http://tfhub.dev), the online catalog of pre-trained models, and use the module `tensorflow-hub` to fetch and instantiate them (experimenting with ready-to-use _Inception_ and _MobileNet_ models).
- 4.4 - [Applying Transfer Learning](./ch4_nb4_apply_transfer_learning.ipynb)
    - Experiment with _transfer learning_, freezing or fine-tuning layers of models pre-trained on different datasets.
- 4.5 - (Appendix) [Exploring ImageNet and Tiny-ImageNet](./ch4_nb5_explore_imagenet_and_its_tiny_version.ipynb)
    - Learn more about _ImageNet_ and _Tiny-ImageNet_, and how to train models on these more complex datasets.
	
## :page_facing_up: Additional Files

- [cifar_utils.py](cifar_utils.py): utility functions for the _CIFAR_ dataset, using `tensorflow-datasets` (code presented in notebook [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [classification_utils.py](classification_utils.py): utility functions for classification tasks, e.g., to load images or to display predictions (code presented in notebook [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): custom Keras _callbacks_ to monitor the trainings of models (code presented in notebook [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [resnet_functional.py](resnet_functional.py): implementation of _ResNet_ using _Keras Functional_ API (code presented in notebook [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [resnet_objectoriented.py](resnet_objectoriented.py): another implementation of _ResNet_ using _Keras Functional_ API, following the _object-orietend_ paradigm (code presented in notebook [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [tiny_imagenet_utils.py](tiny_imagenet_utils.py): utility functions for the _Tiny-ImageNet_ dataset (code presented in notebook [4.5](./ch4_nb5_explore_imagenet_and_its_tiny_version.ipynb)).
