import numpy as np 

"""Skeleton for exercise 4 of exercise sheet 1"""

class ReLU:

    """Implementation of the ReLU activation function."""

    ### add here your implementation of the class ReLU

if __name__ == '__main__':
    
    fun = ReLU()
    x = np.arange(-5, 6, dtype=float)

    print(x)
    print(fun(x))
    """
    Expected output:
    >>> [-5. -4. -3. -2. -1.  0.  1.  2.  3.  4.  5.]
    >>> [0. 0. 0. 0. 0. 0. 1. 2. 3. 4. 5.]  
    """
