"""Test the VanillaRNN layer."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from amllib.layers import VanillaRNN
from amllib.activations import Linear


class TestVanillaRNN(unittest.TestCase):

    def test_evaulation(self):

        n, ts, ni = 1, 2, 10
        no = 4

        test_in = np.arange(n*ts*ni)
        test_in = test_in.reshape(n, ts, ni)
        layer = VanillaRNN(no=no, ni=(ts, ni))
        layer.Wh = 0.5 * np.eye(no)
        layer.Wx = 4 * np.eye(ni, no)
        layer.b = 2 * np.arange(no)
        layer.afun = [Linear() for _ in range(ts)]

        hs_true = np.array([[[0, 6, 12, 18],
                             [40, 49, 58, 67]]])

        hs_pred = layer(test_in)

        assert_array_equal(hs_true, hs_pred, err_msg=f'Predicted output {hs_pred} differs'
                                                     f'from the expected output {hs_true}.')

    def test_feedforward(self):

        n, ts, ni = 1, 2, 10
        no = 4

        test_in = np.arange(n*ts*ni)
        test_in = test_in.reshape(n, ts, ni)
        layer = VanillaRNN(no=no, ni=(ts, ni))
        layer.Wh = 0.5 * np.eye(no)
        layer.Wx = 4 * np.eye(ni, no)
        layer.b = 2 * np.arange(no)
        layer.afun = [Linear() for _ in range(ts)]

        hs_true = np.array([[[0, 6, 12, 18],
                             [40, 49, 58, 67]]])

        hs_pred = layer.feedforward(test_in)

        assert_array_equal(hs_true, hs_pred, err_msg=f'Predicted output {hs_pred} differs'
                                                     f'from the expected output {hs_true}.')

    def test_backprop(self):

        n, ts, ni = 1, 2, 10
        no = 4

        test_in = np.arange(n*ts*ni)
        test_in = test_in.reshape(n, ts, ni)
        layer = VanillaRNN(no=no, ni=(ts, ni))
        layer.Wh = np.eye(no)
        layer.Wx = np.eye(ni, no)
        layer.b = np.zeros(no)
        layer.afun = [Linear() for _ in range(ts)]

        hs = np.arange(n*ts*no).reshape(n, ts, no)

        hs_out = layer.feedforward(test_in)
        delta_out = layer.backprop(hs_out - hs)

        delta_true = np.array([[[6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
                                [6, 7, 8, 9, 0, 0, 0, 0, 0, 0]]])
        dWh_true = np.array([[0, 1, 2, 3]]).T @ np.array([[6, 7, 8, 9]])

        dWx_true = np.arange(10, 20).reshape(
            10, 1) @ np.array([[6, 7, 8, 9]]) + np.arange(10).reshape(10, 1) @ np.array([[6, 7, 8, 9]])

        db_true = np.array([12, 14, 16, 18])

        assert_array_equal(delta_out, delta_true, err_msg=f'Backpropagation output {delta_out} differs'
                                                          f' from the expected output {delta_true}.')
        assert_array_equal(layer.dWh, dWh_true, err_msg=f'Computed cell weight update {layer.dWh} differs'
                                                        f' from the expected update {dWh_true}.')
        assert_array_equal(layer.dWx, dWx_true, err_msg=f'Computed input weight update {layer.dWx} differs'
                                                        f' from the expected update {dWx_true}.')
        assert_array_equal(layer.db, db_true, err_msg=f'Computed bias update {layer.db} differs'
                                                      f' from the expected update {db_true}.')

if __name__ == '__main__':

    unittest.main()
