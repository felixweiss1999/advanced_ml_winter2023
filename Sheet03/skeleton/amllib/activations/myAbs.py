"""Implementation of the Abs activation function."""

__author__ = "Felix Weiß"
__version__ = "1.0"

__name__ = "amllib.activations.myabs"
__package__ = "amllib.activations"

import numpy as np

from .activation import Activation

class myAbs(Activation):
    """
    Class representation of the Abs activation function.

    This class represents the Abs activation function
    $$
    \\text{Abs}(x) = \\begin{cases}x, \\quad
    x \\geq 0 \\\\ 0, \\quad x < 0 \\end{cases}.
    $$

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor: float
        Scaling factor for weight initialization. This factor is shared
        by all Abs like activation functions. It is set to $\\sqrt{2}$.
    """

    def __init__(self):
        """
        Initialize the Abs activation function.
        """
        super().__init__()
        self.name = 'myAbs'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Abs activation function.

        This method applies the Abs activation function
        componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray:
            Output array, has the same shape as the input `x`.
        """

        return np.abs(x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Abs activation function.

        This method applies the derivative of the Abs function
        componentwise to an array.

        **Note**: Since the ReLU function is not differentiable
        in `x = 0`, a weak derivative is used here. The point `0` is
        evaluated to 1.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray:
            Output array, has the same shape as the input `x`.
        """
        return np.sign(x)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Abs activation function.

        This method applies the Abs function
        componentwise to an array. Data is cached
        for later backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------

        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        self.data = x
        return np.abs(x)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the ReLU function and
        multiply the result with the input.

        This method applies the derivative of the ReLU
        function componentwise to the last input of the
        `feedforward` method. The result is then multiplied
        with the input.

        **Note**: Since the ReLU function is not differentiable
        in `x = 0`, a weak derivative is used here. The point
        `0` is evaluated to `1`.

        Parameters
        ----------
        delta : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `delta`.

        Raises
        ------
        ValueError
            Raised if the `feedforward` method was not called before.
        """
        if self.data is None:
            raise ValueError('The feedforward method was not'
                             'called previously. No data'
                             'for backpropagation available')
        # IMPORTANT: this is computing the delta_i vector!!! the self.data attribute is local to a specific LAYER!!!
        # we have one activation function object for every layer!! So this is all local to one layer! Note also that in the
        # slides, the round ° thingy is componentwise multiplication, not some dot product or whatever
        # self.data = z_m-1, delta = a_m - y. These two are vectors of the same dimension!
        # also, the returned vector equals the derivative of the Cost function w.r.t. 
        return np.sign(self.data) * delta
