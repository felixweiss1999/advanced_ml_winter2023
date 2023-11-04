"""
Solution of exercise 2b) of exercise sheet 2.
Test script for the simple feedforward neural network
implementation. The network implementation is tested
with the structure for the logical AND, OR, and XOR
from exercise sheet 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from feedforward import FeedforwardNet

if __name__ == '__main__':

    # Set up the network of exercise 1, exercise sheet 1
    model = FeedforwardNet([2, 4, 3])

    W1 = np.array([[-1., -1.],
                   [-1., 1.],
                   [1., -1.],
                   [1., 1.]])

    W2 = np.array([[0., 0., 0., 1.],
                   [0., 1., 1., 1.],
                   [0., 1., 1., 0.]])

    b1 = np.array([[1.],
                   [0.],
                   [0.],
                   [-1.]])

    model.set_weights(W1, 0)
    model.set_weights(W2, 1)
    model.set_bias(b1, 0)

    # Test the network
    x = np.array([[0., 0., 1., 1.],
                  [0., 1., 0., 1.]])

    print('------------ test the network ------------')
    print('input:')
    print(x)
    print('output:')
    print(model(x))

    print('------------ display the network ---------')
    model.draw()
    plt.show()
