import numpy as np
import unittest
from exercise4 import ReLU

class TestReLU(unittest.TestCase):

    def test_relu(self):

        # initialize test array and desired output
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        s = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0])

        # inititalize ReLU and evaluate __call__ method
        activation = ReLU()
        y = activation(x)

        for i in range(x.size):
            eval_msg = f'{activation.name} should evaluate'\
              f' {x[i]} to {s[i]}, but the result is {y[i]}.'
            self.assertEqual(y[i], s[i], eval_msg)

if __name__ == '__main__':
    unittest.main()
