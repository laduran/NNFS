import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = [2]
output = np.dot(inputs, weights) + bias
print(output)


weights2 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases2 = [2, 3, 0.5]

output2 = np.dot(weights2, inputs) + biases2
print(output2)
