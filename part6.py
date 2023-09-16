# Adds Softmax Activation Function

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # random init of weights and biases
        self.biases = np.zeros((1, n_neurons)) # arg is a tuple
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU():
    '''
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    basically:

        (x + abs(x))/2

    ReLU is better than the sigmoid activation function because:

    Relu does not have vanishing gradient problem.
    Relu is more computationally efficient
    Networks with ReLU tend to show better convergence performance.
    '''
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    '''
    Softmax activiatoin function.
    https://en.wikipedia.org/wiki/Softmax_function
    '''
    def forward(self, inputs):
        # easy to run into overflow issues if we simply did np.exp(inputs) 
        # therefore we subract the max input from all inputs before applying exp
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(n_inputs=3, n_neurons=3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5]) # just print the first five
# Next video (part 7) is to train the model and compute prediction error
# calculation of the loss function
