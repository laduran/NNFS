# Now compute the loss function (on the way to back propagation)
# For classification use cases we use Categorical Cross-Entropy function
# https://en.wikipedia.org/wiki/Cross-entropy
#
# natural log = log(base) e
#
# TODO: UNderstand what the concept of one-hot encoding/vector
import numpy as np
import math

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

softmax_output = [0.7, 0.1, 0.2] # example output from Softmax activation
target_output = [1, 0, 0] # if target class = 0

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])
print(loss)

# same calc
loss = -(math.log(softmax_output[0]))
print(loss)