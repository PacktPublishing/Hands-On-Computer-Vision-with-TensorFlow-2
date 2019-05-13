**> Chapter 1:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# Computer Vision & Neural Networks

In Chapter 1 of the book, we introduced computer vision and machine learning, explaining in details how neural networks work. In this directory, we provide an implementation from scratch, applying our simple network to a historical classification task.

## :notebook: Notebooks

(Reminder: Notebooks are better visualized with `nbviewer`: click [here](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch1) to continue on `nbviewer.jupyter.org`.)

- 1.1 - [Building and Training a Neural Network from Scratch](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)
    - Implement a simple neural network, from the *modelling of an artificial neuron* to a *multi-layered system* which can be trained to classify images of hand-written digits.
	
## :page_facing_up: Additional Files

- [neuron.py](neuron.py): model of an *artificial neuron* able to forward information (code presented in notebook [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
- [fully_connected_layer.py](fully_connected_layer.py): implementation of a functional *layer* grouping several neurons, with methods to optimize its parameters (code presented in notebook [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
- [simple_network.py](simple_network.py): class wrapping everything together into a modular *neural network* model which can be trained and used for various tasks (code presented in notebook [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
