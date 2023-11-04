"""First implementation of an FNN"""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.1'

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from relu import ReLU

class FeedforwardNet:
    """
    Feedforward neural network class.

    This class is a first implementation of a
    Feedforward neural network.

    Attributes
    ----------
    layers: np.ndarray
        Array filled with the number of neurons for
        each layer.
    weights : list[np.ndarray]
        List of weight matrices of the network.
    biases : list[np.ndarray]
        List of biases of the network.
    afuns : list[ReLU]
        List of activation functions for each layer.
    """

    def __init__(self, layers: list[int]) -> None:
        """
        Initialize the Feedforward network.

        Parameters
        ----------
        layers : list[int]
            List of layer sizes. The first entry is the number
            of inputs of the network and the last entry is the
            number of outputs of the network.
        """
        # Initialize network structure
        self.layers = np.array(layers)
        self.weights = [np.zeros((m, n)) for m, n in zip(layers[1:], layers[:-1])]
        self.biases = [np.zeros((m, 1)) for m in layers[1:]]
        self.afuns = [ReLU() for _ in layers[1:]]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the network.

        For each layer compute the affine linear combination
        with the corresponding weight matrix and the bias, and
        activate the result.

        Parameters
        ----------
        x : np.ndarray
            Input for the network.

        Returns
        -------
        np.ndarray
            Activated output of the last layer.
        """

        ######################################################
        # TODO Implement the feedforward evaluation.         #
        # Iterate through each layer, compute the            #
        # affine linear combination and activate the result. #
        ######################################################
        for i in range(0, len(self.layers)-1):
             x = self.afuns[i](self.weights[i] @ x + self.biases[i])
        return x

    def set_weights(self, W: np.ndarray, index: int) -> None:
        """
        Set the weight matrix of a layer.

        Set the weight matrix of layer `index`.

        Parameters
        ----------
        W : np.ndarray
            Source weight matrix.
        index : int
            Index of the layer.

        Raises
        ------
        ValueError
            Raised if the index is out of bounds or the shape
            of the new weight matrix does not match the
            layer sizes.
        """
        if not index < len(self.weights):
            raise ValueError("Index out of bounds!")
        if not self.weights[index].shape == W.shape:
            raise ValueError("The shape of the new weight matrix "
                             "does not match the size of the layers. "
                             f"It should be {self.weights[index].shape}, "
                             f"but is {W.shape}.")

        self.weights[index] = W

    def set_bias(self, b: np.ndarray, index: int) -> None:
        """
        Set the bias of a layer.

        Set the bias of layer `index`.

        Parameters
        ----------
        b : np.ndarray
            Source bias.
        index : int
            Index of the layer.

        Raises
        ------
        ValueError
            Raised if the index is out of bounds or the shape
            of the new weight matrix does not match the
            layer sizes.
        """

        if not index < len(self.biases):
            raise ValueError("Index out of bounds!")
        if not self.biases[index].shape == b.shape:
            raise ValueError("The shape of the new bias "
                             "does not match the size of the layer."
                             f"It should be {self.biases[index].shape}, "
                             f"but is {b.shape}")

        self.biases[index] = b

    def draw(self, file_name: Optional[str] = None) -> None:
        """
        Draw the network.

        Each layer is drawn as a vertical line of circles
        representing the neurons of this layer.

        Parameters
        ----------
        file_name : str | None
            If `file_name` is not `None`, the image
            is written to a corresponding pdf file.
            Otherwise it is just displayed.
        """

        num_layers = len(self.layers)
        max_neurons_per_layer = np.amax(self.layers)
        dist = 2 * max(1, max_neurons_per_layer / num_layers)
        y_shift = self.layers / 2 - .5
        rad = .3

        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        # Draw all circles
        for i in range(num_layers):
            for j in range(self.layers[i]):
                circle = plt.Circle((i * dist, j - y_shift[i]),
                                    radius=rad, fill=False)
                ax.add_patch(circle)

        # Draw the lines between the layers.
        for i in range(num_layers-1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i+1]):
                    angle = \
                      np.arctan((j - k + y_shift[i+1] - y_shift[i]) \
                                / dist)
                    x_adjust = rad * np.cos(angle)
                    y_adjust = rad * np.sin(angle)
                    line = plt.Line2D((i * dist + x_adjust,
                                       (i+1) * dist - x_adjust),
                                      (j - y_shift[i] - y_adjust,
                                       k - y_shift[i+1] + y_adjust),
                                      lw = 2 / np.sqrt(self.layers[i]
                                                       + self.layers[i+1]),
                                      color='b')
                    ax.add_line(line)

        ax.axis('scaled')

        if file_name is None:
            plt.show()
        else:
            fig.savefig(file_name, bbox_inches='tight', format='pdf')


#Test of this implementation:

w_1 = np.array([[-1, -1],\
                [-1,  1],\
                [ 1, -1],\
                [ 1,  1]])
b_1 = np.array([[ 1],\
               [ 0],\
               [ 0],\
               [-1]])

instance = FeedforwardNet([2, 4, 1])
instance.set_weights(w_1, 0)
instance.set_bias(b_1, 0)
instance.set_weights(np.array([[0,1,1,0]]), 1)
instance.set_bias(np.array([[0]]), 1)
print(instance([[1],[0]]))
#verdict: works, but dimensionality of all involved variables needs to be more carefully considered!