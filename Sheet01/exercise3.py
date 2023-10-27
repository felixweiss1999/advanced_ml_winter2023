import numpy as np
from time import time

"""Solution to exercise 3 of exercise sheet 1"""

np.__config__.show()

n, k = 10000, 100

A = np.random.random((n, n))
B = np.random.random((n, k))
C = np.zeros((n, k))

#####################
# BLAS level 1 test #
#####################
t = time()
for j in range(k):
    for i in range(n):
        C[i, j] = A[i, :] @ B[:, j]

level1_time = time() - t
print(f'BLAS LEVEL 1: {level1_time} seconds')

#####################
# BLAS level 2 test #
#####################
t = time()
for j in range(k):
    C[:, j] = A @ B[:, j]

level2_time = time() - t
print(f'BLAS LEVEL 2: {level2_time} seconds')

#####################
# BLAS level 3 test #
#####################
t = time()
C = A @ B

level3_time = time() - t
print(f'BLAS LEVEL 3: {level3_time} seconds')

print(f'speedup 1 vs. 2: {level1_time / level2_time}')
print(f'speedup 2 vs. 3: {level2_time / level3_time}')
print(f'speedup 1 vs. 3: {level1_time / level3_time}')
