import numpy as np

"""Addendum to exercise 1 of exercise sheet 1"""

# define all four inputs as columnwise data matrix
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
print('data matrix X:')
print(X)

print('--------- part a): AND ---------')

Y = np.array([[0, 0, 0, 1]])
print('desired output Y:')
print(Y)

W = np.array([[1, 1]])
print('weight matrix W:')
print(W)

b = np.array([-1])
print('bias vector b:')
print(b)

Z = W @ X + b
print('affine linear combination Z:')
print(Z)

A = Z.clip(0)
print('activation A:')
print(A)

input('press <ENTER> to continue:')

print('--------- part b): all ---------')

Y = np.array([[0, 0, 0, 1],
              [0, 1, 1, 1],
              [0, 1, 1, 0]])
print('desired output Y:')
print(Y)

W1 = np.array([[-1, -1],
               [-1, 1],
               [1, -1],
               [1, 1]])
print('weight matrix W1:')
print(W1)

b1 = np.array([[1],
               [0],
               [0],
               [-1]])
print('bias vector b1:')
print(b1)

W2 = np.array([[0, 0, 0, 1],
               [0, 1, 1, 1],
               [0, 1, 1, 0]])
print('weight matrix W2:')
print(W2)

b2 = np.zeros((3, 1), dtype=int)
print('bias vector b2:')
print(b2)

input('press <ENTER> to continue:')

A1 = X
print('activation A1:')
print(A1)

Z1 = W1 @ A1 + b1
print('affine linear combination Z1:')
print(Z1)

A2 = Z1.clip(0)
print('activation A2:')
print(A2)

input('press <ENTER> to continue:')

Z2 = W2 @ A2 + b2
print('affine linear combination Z2:')
print(Z2)

A3 = Z2.clip(0)
print('activation A3:')
print(A3)

# just for comparison
print('desired output Y:')
print(Y)
