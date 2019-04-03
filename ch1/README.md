# Chapter 1: Computer Vision & Deep Learning

In Chapter 1 of the book, we introduced computer vision and machine learning, explaining in details how neural networks work. In this directory, we provide an implementation from scratch, applying our simple network to a historical classification task.

## Notebooks

- 1.1 - [Building and Training a Neural Network from Scratch](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)
    - Implement a simpleneural network, from the modelling of an artificial neuron to a multi-layered system which can be trained to classify images of hand-written digits.
	
## Additional Files


- [neuron.py](neuron.py): model of an *artificial neuron* able to forward information (code presented in notebook [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
- [fully_connected_layer.py](fully_connected_layer.py): implementation of a functional *layer* grouping several neurons, with methods to optimize its parameters (code presented in notebook [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
- [simple_network.py](simple_network.py): class wrapping everything together into a modular *neural network* model which can be trained and used for various tasks (code presented in notebook [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).