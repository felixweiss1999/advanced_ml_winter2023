import numpy as np

"""Addendum to exercise 2 of exercise sheet 1"""

# define all four inputs as columnwise data matrix
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
print('data matrix X:')
print(X)

Y = np.array([[0, 1, 1, 0]])
print('desired output Y:')
print(Y)

W = np.array([[1, -1]])
print('weight matrix W:')
print(W)

b = np.array([0])
print('bias vector b:')
print(b)

Z = W @ X + b
print('affine linear combination Z:')
print(Z)

A = np.abs(Z)
print('activation A:')
print(A)
