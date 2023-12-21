"""Test the SoftMaxRNN layer."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from amllib.layers import SoftMaxRNN
from amllib.activations import Linear

class TestSoftMaxRNN(unittest.TestCase):

    def test_evaluation(self):

        n, ts, ni = 1, 2, 4
        no = 3

        test_in = np.arange(n*ts*ni)
        test_in = test_in.reshape(n, ts, ni)

        layer = SoftMaxRNN(no, input_shape=(ts, ni))
        layer.W = np.arange(ni*no).reshape(ni, no)
        layer.b = np.arange(no)

        z_true = np.array([[[42, 49, 56],
                            [114, 137, 160]]])

        e = np.exp(z_true)
        a_true = e / np.sum(e, axis=-1, keepdims=True)

        a_out = layer(test_in)

        assert_array_equal(a_true, a_out, err_msg=f'Predicted output {a_out} differs'
                                                  f'from the expected output {a_true}.')

    def test_feedforward(self):

        n, ts, ni = 1, 2, 4
        no = 3

        test_in = np.arange(n*ts*ni)
        test_in = test_in.reshape(n, ts, ni)

        layer = SoftMaxRNN(no, input_shape=(ts, ni))
        layer.W = np.arange(ni*no).reshape(ni, no)
        layer.b = np.arange(no)

        z_true = np.array([[[42, 49, 56],
                            [114, 137, 160]]])

        e = np.exp(z_true)
        a_true = e / np.sum(e, axis=-1, keepdims=True)

        a_out = layer.feedforward(test_in)

        assert_array_equal(a_true, a_out, err_msg=f'Predicted output {a_out} differs'
                                                  f'from the expected output {a_true}.')

    def test_backprop(self):

        n, ts, ni = 1, 2, 4
        no = 3

        test_in = np.arange(n*ts*ni)
        test_in = test_in.reshape(n, ts, ni)

        layer = SoftMaxRNN(no, input_shape=(ts, ni))
        layer.W = np.arange(ni*no).reshape(ni, no)
        layer.b = np.arange(no)

        delta = 2 * np.arange(n*ts*no).reshape(n, ts, no)

        layer.feedforward(test_in)
        delta_out = layer.backprop(delta)

        delta_true = np.array([[[10, 28, 46, 64],
                                [28, 100, 172, 244]]])

        dW_true = np.array([[24, 32, 40],
                            [30, 42, 54],
                            [36, 52, 68],
                            [42, 62, 82]])

        db_true = np.array([6, 10, 14])

        assert_array_equal(delta_out, delta_true, err_msg=f'Backpropagation output {delta_out} differs'
                                                          f' from the expected output {delta_true}.')
        assert_array_equal(layer.dW, dW_true, err_msg=f'Computed weight matrix update {layer.dW} differs'
                                                      f' from the expected update {dW_true}.')
        assert_array_equal(layer.db, db_true, err_msg=f'Computed bias update {layer.db} differs'
                                                      f' from the expected update {db_true}.')

if __name__ == '__main__':

    unittest.main()
