
__name__ = 'amllib.regularizer.l2regularizers'
__package__ = 'amllib.regularizers'

from .regularizer import Regularizer
import numpy as np

class L2Regularizer(Regularizer):
    """
    Class representation of the L2 regularization.

    This class provides the regularization of parameter
    updates with the L2 regularization (see lecture notes, section 2.6.1).

    Attributes
    ----------
    l2 : float
        Regularization parameter.
    """
    def __init__(self, l2: float):
        """
        Initialize the L2 regularizer.

        Parameters
        ----------
        l2 : float
            Regularization parameter.
        """

        self.l2 = l2

    def regularize(self, ddata: np.ndarray, rdata: np.ndarray) -> None:
        """
        Regularize update data for parameter.

        Regularizes the update data in ddata with data
        by L2 regularization. data is scaled by the
        parameter l2 and then added to the update data.


        Parameters
        ----------
        ddata : np.ndarray
            Regularization data.
        rdata : np.ndarray
            Parameter update data.
        """

        # TODO Implement the L2 regularization of parameter updates.
        # This method is used to obtain the gradients of the
        # cost function with penalty term with backpropagation
        # (See lecture notes, section 2.6.1).
