"""Implementation of an Elman RNN"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.layers.rnn"
__package__ = "amllib.layers"

from typing import Optional, Union

import numpy as np

from ..initializers import RandnOrthonormal, RandnAverage
from ..optimizers import Optimizer, SGD
from ..activations import TanH, SoftMax
from .layer import Layer


class VanillaRNN(Layer):
    """
    Class representation of the hidden state part
    of an Elman RNN layer.

    This class represents the hidden state part of an
    Elman RNN layer. From the previous hidden state and
    the input a new hidden state is computed as
    activated affine linear combination.

    Attributes
    ----------
    no : int
        Number of outputs per sequence member.
    ni : int
        Number of inputs per sequence member.
    max_t : int
        Maximal sequence length.
    bs : int
        Batch size of the inputs.
    use_clipping : bool
        True if the gradients should be clipped
        while updating the weights.
    optim : Optimizer
        Optimizer for training
    recurrent_initializer : RandnOrthonormal
        Initializer for the hidden state weight matrix.
    kernel_initializer : RandnAverage
        Initializer for the input weight matrix.
    afun : list[TanH]
        List of activation functions. One for each possible
        input sequence member.
    h : np.ndarray
        Last computed cell state, filled with zeros on initialization.
    __a : np.ndarray
        Last computed affine linear combination.
    x : np.ndarray
        Last input sequence, initialized as None.
    y : np.ndarray
        Last computed sequence of cell states,
        initialized as None.
    Wh : np.ndarray
        Weight matrix for the hidden state. It has shape `(no, no)`.
    Wx : np.ndarray
        Weight matrix for the input sequence. It has shape `(ni, no)`.
    b : np.ndarray
        Bias vector. It has shape `(no,)`.
    dWh : np.ndarray
        Storage for the update of the hidden state weight matrix `Wh`.
    dWx : np.ndarray
        Storage for the update of the input sequence weight matrix `Wx`.
    db : np.ndarray
        Storage for the update of the bias.
    """

    def __init__(self, no: int,
                 ni: Optional[tuple[int, int]] = None,
                 optim: Optional[Optimizer] = None,
                 stateful: bool = False,
                 use_clipping: bool = True):
        """
        Initialize the Elman RNN layer

        Parameters
        ----------
        no : int
            Output dimension.
        ni : tuple[int, int], optional
            Input dimension, by default None. If it is
            not None the layer is built on initialization.
        optim : Optimizer, optional
            Optimizer for training, by default SGD(0.01) with momentum 0.9.
        use_clipping : bool, optional
            True if the gradients should be clipped
            while updating the weights, by default True
        """
        super().__init__()

        self.no = no
        self.use_clipping = use_clipping
        self.stateful = stateful
        if optim is None:
            self.optim = SGD(eta=.01, momentum=.9)
        else:
            self.optim = optim

        # Set initializer for Input data weights.
        self.recurrent_initializer = RandnOrthonormal()
        # Set initializer for cell state weights.
        self.kernel_initializer = RandnAverage()

        # Allocate memory for the cell state
        # Last computed cell state. This is reset
        # at each call of feedforward.
        self.h = None
        self.__a = None

        self.x = None
        self.y = None  # Last computed sequence of cell states

        self.ni = None
        self.max_t = None

        if ni is not None:
            self.build(ni)

    def build(self, input_shape: tuple[int, int]) -> None:
        """
        Build the layer.

        The input shape is set and the weight matrix, as well as the
        bias are initialized.

        Parameters
        ----------
        input_shape : Union[int, tuple[int]]
            Shape of the inputs (without sequence and batch dimension).
        """

        (self.max_t, self.ni) = input_shape

        # Set activation functions
        self.afun = [TanH() for t in range(self.max_t)]

        # Initialize weight matrices.
        self.Wh = self.recurrent_initializer.wfun(self.no, self.no)
        self.Wx = self.kernel_initializer.wfun(self.ni, self.no)

        # Initialize bias.
        self.b = np.zeros(self.no)

        # Allocate memory for updates.
        self.dWh = np.zeros_like(self.Wh)
        self.dWx = np.zeros_like(self.Wx)
        self.db = np.zeros_like(self.b)

        self.built = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the RNN layer.

        Parameters
        ----------
        x : np.ndarray
            Input sequence.

        Returns
        -------
        np.ndarray
            Sequence of cell states
        """

        # If the layer is not built yet, do so with the shape of the input data.
        n, ts, ni = x.shape
        if not self.built:
            self.build((ts, ni))

        y = np.zeros((n, ts+1, self.no))
        a = np.zeros((n, self.no))

        if self.h is None or not self.stateful:
            h = np.zeros((n, self.no))
        else:
            h = self.h.mean(axis=0)*np.ones((n, self.no))

        # Initialize the output sequence
        y[:, 0, :] = h

        # TODO implement the feedforward for the cell states of
        # an Elman RNN. Iterate through all time steps and
        # compute the cell states from the previous one.

        return y[:, 1:, :]

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the RNN layer.

        Evaluate the RNN layer and cache data for backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input sequence.

        Returns
        -------
        np.ndarray
            Sequence of cell states
        """
        self.x = x

        # If the layer is not built yet, do so with the shape of the  input data.
        n, ts, ni = x.shape
        if not self.built:
            self.build((ts, ni))

        self.__a = np.zeros((n, ts, self.no))

        if self.h is None or not self.stateful:
            self.h = np.zeros((n, self.no))

        self.y = np.zeros((n, ts+1, self.no))
        self.y[:, 0, :] = self.h

        # TODO implement the feedforward for the cell states of
        # an Elman RNN. Iterate through all time steps and
        # compute the cell states from the previous one.
        # Instead of self.afun[t] use self.afun[t].feedforward
        # for activation. This method caches all necessary data
        # for the backpropagation.

        return self.y[:, 1:, :]

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through this layer to obtain updates for the
        weight matrices.

        Computes the derivative of a cost function by the weight
        matrices, the bias and the input from a given derivative
        by the output. The derivatives are obtained from applying
        the chain rule.

        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the output.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """

        n, ts, no = delta.shape
        dx = np.zeros((n, ts, self.ni))
        dh_next = np.zeros((n, no))

        self.dWh.fill(0)
        self.dWx.fill(0)
        self.db.fill(0)

        # TODO Implement the backpropagation through time
        # for Elman RNNs. See the lecture notes, section 5.4
        # for more information.

        return dx

    def update(self):
        """
        Update the weight matrices and the bias.

        The updates computed by the backpropagation
        method are applied with the optimizer defined
        on initialization.
        """

        if self.use_clipping:
            for param in [self.dWh, self.dWx, self.db]:
                np.clip(param, -4., 4., out=param)

        self.optim.update([self.Wh, self.Wx, self.b],
                          [self.dWh, self.dWx, self.db])

    def get_output_shape(self) -> tuple[int]:
        """
        Get the shape of the outputs (without the sequence and batch dimension).

        Returns
        -------
        tuple[int]
            Output shape
        """
        return (self.max_t, self.no)


class SoftMaxRNN(Layer):
    """
    Class representation of the SoftMax part of
    an Elman RNN cell

    This class represents the second part of an Elman
    RNN cell. The hidden state computed by the `VanillaRNN`
    layer gets activated with the SoftMax activation function,
    after applying an affine linear combination.

    Attributes
    ----------
    no : int
        Number of outputs per sequence member.
    ni : int
        Number of inputs per sequence member.
    max_t : int
        Maximal sequence length.
    use_clipping : bool
        True if the gradients should be clipped
        while updating the weights.
    afun : SoftMax
        Activation function. It is set up as an instance
        of the `SoftMax` class on initialization.
    optim : Optimizer
        Optimizer for training.
    kernel_initializer : RandnAverage
        Initializer for the weight matrix. It is set up
        as an instance of the `RandnAverage` class on
        initialization.
    a : np.ndarray
        Last computed activated output.
    __z : np.ndarray
        Last computed affine linear combination.
    W : np.ndarray
        Weight matrix of shape `(ni, no)`.
    b : np.ndarray
        Bias of shape `(no,)`.
    dW : np.ndarray
        Storage for the update of the weight matrix.
    db : np.ndarray
        Storage for the update of the bias.
    """

    def __init__(self, no: Union[int, tuple[int]],
                 input_shape: Union[int, tuple[int]] = None,
                 optim: Optimizer = None, use_clipping: bool = True) -> None:
        """
        Initialize the SoftMaxRNN layer.

        Parameters
        ----------
        no : int | tuple[int]
            Output shape | Number of input sequence members.
        input_shape : tuple[int, int] optional
            Shape of the inputs (without sequence and batch dimension).
            If it is not None, the build method is called with it.
        optim : Optimizer, optional
            Optimizer for training, by default SGD(0.01).
        use_clipping : bool, optional
            True if the gradients should be clipped
            while updating the weights, by default True
        """
        super().__init__()

        self.no = no
        self.use_clipping = use_clipping

        self.afun = SoftMax()

        if optim is None:
            self.optim = SGD(eta=.01, momentum=.9)
        else:
            self.optim = optim

        self.kernel_initializer = RandnAverage()

        self.ni = None
        self.max_t = None

        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape: Union[int, tuple[int]]) -> None:
        """
        Build the SoftMaxRNN layer.

        The input shape is set and the weight matrix,
        as well as the bias are initialized.

        Parameters
        ----------
        input_shape : Union[int, tuple[int]]
            Shape of the inputs (without sequence and batch dimension).
        """

        (self.max_t, self.ni) = input_shape

        self.W = self.kernel_initializer.wfun(self.ni, self.no)
        self.b = np.zeros(self.no)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.a = None

        self.built = True

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the SoftMaxRNN layer.

        The hidden state sequence given as argument
        is combined with the weight matrix and the bias
        and then activated with the SoftMax activation
        function.

        Parameters
        ----------
        a : np.ndarray
            Input sequence. This should be the sequence
            of hidden states computed by a VanillaRNN layer.

        Returns
        -------
        np.ndarray
            Activated output sequence.
        """

        n, t, ni = a.shape
        if not self.built:
            self.build((t, ni))

        # TODO Implement the call for the outputs of an Elman RNN.
        # For each time step compute the affine linear combination of the
        # input a[:,t,:] with the weights and the bias.
        # HINTS:
        #  - You can use np.transpose to swap the time and the batch axis.
        #  - The numpy.dot function computes the matrix product over the last two axes
        #    of the factors, e.g. a @ self.W computes a[i,:,:] @ W for all i.

        z = None
        return self.afun(z)

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the SoftMaxRNN layer.

        The hidden state sequence given as argument
        is combined with the weight matrix and the bias
        and then activated with the SoftMax activation
        function. Data is cached for backpropagation.

        Parameters
        ----------
        a : np.ndarray
            Input sequence. This should be the sequence
            of hidden states computed by a VanillaRNN layer.

        Returns
        -------
        np.ndarray
            Activated output sequence.
        """

        n, t, ni = a.shape
        if not self.built:

            self.build((t, ni))

        self.a = a

        # TODO Implement the feedforward for the outputs of an Elman RNN.
        # For each time step compute the affine linear combination of the
        # input a[:,t,:] with the weights and the bias.
        # HINTS:
        #  - You can use np.transpose to swap the time and the batch axis.
        #  - The numpy.dot function computes the matrix product over the last two axes
        #    of the factors, e.g. a @ self.W computes a[i,:,:] @ W for all i.

        z = None
        return self.afun.feedforward(z)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Propagate through the layer to compute the updates
        of the weights and biases.

        Backpropagation through time is used to compute
        updates for the weight matrix and the bias. The computations
        are based on the derivative of the cost function
        by the output of this layer, which is given as argument.
        The dervative of the cost function by the inputs
        is returned.

        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the outputs.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the inputs.
        """

        n, ts, ni = delta.shape

        self.dW.fill(0)
        self.db.fill(0)

        delta = self.afun.backprop(delta)

        # TODO Implement the backpropagation through time
        # for Elman RNNs. See the lecture notes, section 5.4
        # for more information.
        # delta_new should contain W_{yh} \delta_{t} for all t.
        # HINTS:
        #  - You can use np.transpose to swap the time and the batch axis.
        #  - The numpy.dot function computes the matrix product over the last two axes
        #    of the factors, e.g. a @ self.W computes a[i,:,:] @ W for all i.

        delta_new = None
        return delta_new

    def update(self) -> None:
        """
        Update the parameters of the SoftMaxRNN layer.

        The weight matrix and the bias are updated with
        the optimizer defined on initialization.
        """
        if self.use_clipping:
            for param in [self.dW, self.db]:
                np.clip(param, -4., 4., out=param)

        self.optim.update([self.W, self.b],
                          [self.dW, self.db])

    def get_output_shape(self) -> tuple[int, int]:
        """
        Get the shape of the outputs of the layer.

        The shape of the outputs (without the
        sequence and the batch dimension) is returned.

        Returns
        -------
        tuple[int, int]
            Output sequence shape.
        """
        return (self.max_t, self.no)


class SimpleRNN(Layer):
    """
    This class represents an Elman RNN layer.

    The representation contains several `VanillaRNN` layers
    followed by a SoftMaxRNN layer.

    Attributes
    ----------
    vanilla_rnn_layers : list[VanillaRNN]
        List of `VanillaRNN` layers, each with an
        hidden state.
    softmax_rnn_layer : SoftMaxRNN
        Final SoftMaxRNN layer
    """

    def __init__(self, cell_state_shape: Union[int, tuple[int]],
                 output_shape: Union[int, tuple[int]],
                 num_vanilla_layers: int = 1,
                 input_shape: Optional[tuple[int, int]] = None,
                 stateful: bool = False,
                 cell_state_optims: list[Optimizer] = None,
                 softmax_optim: Optimizer = None,
                 use_clipping: bool = True) -> None:
        """
        Initialize the `SimpleRNN` layer.

        Parameters
        ----------
        cell_state_shape : int | tuple[int]
            Output shape for the `VanillaRNN` layers.
        output_shape : int | tuple[int]
            Output shape for the `SoftMaxRNN` layer.
        max_t : int
            Maximal sequence length.
        num_vanilla_layers : int
            Number of `VanillaRNN` layers, by default 1.
        input_shape : int | tuple[int], optional
            Shape of the inputs to this layer.
        cell_state_optim : Optimizer
            Optimizer for the `VanillaRNN` layers, by default SGD(0.01).
            See `VanillaRNN` for more information.
        softmax_optim : Optimizer, optional
            Optimizer for the `SoftMaxRNN` layer, by default SGD(0.01).
            See `SoftMaxRNN` for more information.
        use_clipping : bool
            Flag indicating if the updates get clipped to [-4,4], by default True
        """

        if num_vanilla_layers == 0:
            self.vanilla_rnn_layers = []
        else:
            self.vanilla_rnn_layers = [VanillaRNN(cell_state_shape,
                                                  stateful=stateful,
                                                  optim=cell_state_optims[0],
                                                  use_clipping=use_clipping)] +\
                [VanillaRNN(cell_state_shape,
                            stateful=stateful,
                            optim=cell_state_optims[i],
                            use_clipping=use_clipping)
                 for i in range(1, num_vanilla_layers)]

        self.softmax_rnn_layer = SoftMaxRNN(output_shape,
                                            optim=softmax_optim,
                                            use_clipping=use_clipping)

        self.built = False

        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape: Union[int, tuple[int]]) -> None:
        """
        Build the `SimpleRNN` layer.

        All layers in `self.vanilla_rnn_layers`
        and `self.softmax_rnn_layer` are built.

        Parameters
        ----------
        input_shape : int | tuple[int]
            Shape of the inputs
        """

        self.vanilla_rnn_layers[0].build(input_shape)
        for i in range(1, len(self.vanilla_rnn_layers)):
            self.vanilla_rnn_layers[i].build(
                self.vanilla_rnn_layers[i-1].get_output_shape()
            )
        self.softmax_rnn_layer.build(
            self.vanilla_rnn_layers[-1].get_output_shape())

        self.built = True

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the `SimpleRNN` layer.

        All `VanillaRNN` layers and the `SoftMaxRNN` layer
        are evaluated consecutively.

        Parameters
        ----------
        input : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Activated output of `self.softmax_rnn_layer`
        """

        h = input
        for layer in self.vanilla_rnn_layers:
            h = layer(h)
        y = self.softmax_rnn_layer(h)

        return y

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the `SimpleRNN` layer.

        All `VanillaRNN` layers and the `SoftMaxRNN` layer
        are evaluated consecutively. Data is cached for later
        backpropagation.

        Parameters
        ----------
        input : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Activated output of `self.softmax_rnn_layer`
        """

        h = input
        for layer in self.vanilla_rnn_layers:
            h = layer.feedforward(h)
        y = self.softmax_rnn_layer.feedforward(h)

        return y

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Propagate backwards through the `SimpleRNN` layer.

        First the `backprop` method of `self.softmax_rnn_layer`
        is called, then (in reversed order) consecutively the
        `backprop` method of the layers contained in
        `self.vanilla_rnn_layers`.

        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the outputs.
        Returns
        -------
        np.ndarray
             Derivative of the cost function by the inputs.
        """

        dy = self.softmax_rnn_layer.backprop(delta)
        dh = dy
        for layer in reversed(self.vanilla_rnn_layers):
            dh = layer.backprop(dh)

        return dh

    def get_output_shape(self) -> tuple[int]:
        """
        Get the shape of the layers outputs.

        Returns
        -------
        tuple[int]
            Output shape as 1-tuple.
        """

        return self.softmax_rnn_layer.get_output_shape()

    def update(self):
        """
        Update the parameters of the contained
        `VanillaRNN` layers and the `SoftMaxRNN`
        layer.
        """

        for layer in self.vanilla_rnn_layers:
            layer.update()
        self.softmax_rnn_layer.update()
