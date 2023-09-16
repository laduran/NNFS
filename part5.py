# This part goes over activation functions
# ReLU, Step, Sigmoid and 
# Will not include SoftMax function
# most popular activation function to use is ReLU

import numpy as np
import nnfs

nnfs.init()

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [1, 2, -1, 3.3, 2.7, 1.1, 2.2, -100]
output = []


def create_data(points, classes, dimensionality=2):
    ''' 
    This function is effectively the same dataset as nnfs.datasets.spiral_data
    Originally from Stanford Machine learning class on convolutional neural networks
    '''
    X = np.zeros((points*classes, dimensionality)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for j in range(points):
        ix = range(classes*j,points*(j+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # random init of weights and biases
        self.biases = np.zeros((1, n_neurons)) # arg is a tuple
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)

activation1.forward(layer1.output)
print(layer1.output)

