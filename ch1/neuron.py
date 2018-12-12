import numpy as np


class Neuron(object):
    """
    A simple artificial neuron, processing an input vector and returning a corresponding activation.
    Args:
        num_inputs (int): The input vector size / number of input values.
        activation_function (callable): The activation function defining this neuron.
    Attributes:
        W (ndarray): The weight values for each input.
        b (float): The bias value, added to the weighted sum.
        activation_function (callable): The activation function computing the neuron's output.
    """

    def __init__(self, num_inputs, activation_function):
        super().__init__()

        # Randomly initializing the weight vector and the bias value:
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)

        self.activation_function = activation_function

    def forward(self, x):
        """
        Forward the input signal through the neuron, returning its activation value.
        Args:
            x (ndarray): The input vector, of shape `(1, num_inputs)`
        Returns:
            activation (ndarray): The activation value, of shape `(1, layer_size)`.
        """
        z = np.dot(x, self.W) + self.b
        return self.activation_function(z)


# Demonstrating how to use the Neuron:
if __name__ == "__main__":
    np.random.seed(42)      # Fixing the seed for the random number generation, to get reproducable results.

    x = np.random.rand(3).reshape(1, 3)   # Random input column array of 3 values (shape = `(1, 3)`)
    # > [[0.37454012 0.95071431 0.73199394]]

    # Instantiating a Perceptron (simple neuron with step function):
    step_function = lambda y: 0 if y <= 0 else 1

    perceptron = Neuron(num_inputs=x.size, activation_function=step_function)
    # > perceptron.W    = [0.59865848 0.15601864 0.15599452]
    # > perceptron.b    = [0.05808361]

    out = perceptron.forward(x)
    # > 1

