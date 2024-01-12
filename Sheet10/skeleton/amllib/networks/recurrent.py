"""Implementation of a recurrent neural network."""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.1'

__name__ = 'amllib.networks.recurrent'
__package__ = 'amllib.networks'

import numpy as np

from amllib.layers import Layer, VanillaRNN, SoftMaxRNN

class Recurrent:
    """
    Class representing a recurrent neural network.
    """

    def __init__(self, input_shape: tuple[int, ...]):
        """
        Initialize the recurrent neural network.

        Parameters
        ----------
        input_shape : tuple[int, ...]
            Input shape (without the batch dimension).
        """

        self.layers = []
        self.input_shape = input_shape
        self.output_shape = input_shape

    def __call__(self, input : np.ndarray) -> np.ndarray:
        """
        Evaluate the recurrent network layer-wise.

        Each layer in the layers list is evaluated with
        the output of the last layer as input.

        Parameters
        ----------
        input : np.ndarray
            Input for the first layer.

        Returns
        -------
        np.ndarray
            Result of the evaluation of the last layer in
            the layers list.
        """

        x = input
        for layer in self.layers:
            x = layer(x)

        return x      

    def feedforward(self,  input: np.ndarray) -> np.ndarray:
        """
        Evaluate the recurrent network layer-wise while training.

        Each layer in the layers list is evaluated with
        the output of the last layer as input.

        Parameters
        ----------
        input : np.ndarray
            Input for the first layer.

        Returns
        -------
        np.ndarray
            Result of the evaluation of the last layer in
            the layers list.
        """

        x = input
        for layer in self.layers:
            x = layer.feedforward(x)

        return x

    def backprop(self, input: np.ndarray, labels: np.ndarray) -> None:
        """
        Backpropate backwards through the list of layers.

        Go backwards through the list of layers and compute
        the updates for all parameters (weights/filter banks and biases).
        For this the backprop method of each layer is used.

        Parameters
        ----------
        input : np.ndarray
            Input data for the backpropagation.
        labels : np.ndarray
            Labels for the input data.

        Raises
        ------
        ValueError
            Raised if the input data shape does not match the input shape
            of the network.
        ValueError
            Raised if the labels shape does not match the output shape of
            the network.
        """

        if input.shape[1:] != self.input_shape:
            raise ValueError("The input shape does not match the output shape of the network!"
                             f"Shape {self.input_shape} was expected, but it was {input.shape[1:]}")
        
        if labels.shape[1:] != self.output_shape:
            raise ValueError("The labels shape does not match the output shape of the network!"
                             f"Shape {self.output_shape} was expected, but it was {labels.shape[1:]}")

        # Compute derivative of the cost function by the output of the network as
        # intial input for the backpropagation method of the last layer.
        delta = (self.feedforward(input) - labels) / labels.shape[0]

        for layer in reversed(self.layers):

            delta = layer.backprop(delta)

    def train(self, train_input: np.ndarray, train_labels: np.ndarray) -> None:
        """
        Train the recurrent network.

        The network training works directly with batches,
        the epochs and possible permutations have to be carried
        out in the calling routine.

        Parameters
        ----------
        train_input : np.ndarray
            Input data for the training.
        train_labels : np.ndarray
            Labels for each input.
        """

        self.backprop(train_input, train_labels)

        for layer in self.layers:
            layer.update()

    def add_rnn_layer(self, shape, optim=None, stateful=True, use_clipping=True):
        """
        Add a VanillaRNN layer to the list of layers.

        A VanillaRNN layer is added to the list of layers
        and the output shape of the network is
        updated.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the output, usually (t_max, no)
        stateful : bool
            Whether the last hidden state should be used in the
            next step, default: True
        optim : optimizer
            The optimizer used for the training
        use_clipping : bool
            Whether to use clipping in training, default: True
        """

        self.layers.append(VanillaRNN(shape,
                                      self.output_shape,
                                      optim=optim,
                                      stateful=stateful,
                                      use_clipping=use_clipping))
        self.output_shape = (self.output_shape[0], shape)

    def add_classifier(self, shape, optim=None, use_clipping=True):
        """
        Add a SoftMaxRNN layer to the list of layers.

        A SoftMaxRNN layer is added to the list of layers
        and the output shape of the network is
        updated.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the output, usually (t_max, no)
        optim : optimizer
            The optimizer used for the training
        use_clipping : bool
            Whether to use clipping in training, default: True
        """

        self.layers.append(SoftMaxRNN(shape,
                                      self.output_shape,
                                      optim=optim,
                                      use_clipping=use_clipping))
        self.output_shape = (self.output_shape[0], shape)
