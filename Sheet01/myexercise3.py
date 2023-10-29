#b): 
import numpy as np
#np.__config__.show()
#answer: openblas64

#c
import time
n=10000
k=100
A = np.random.random(size=(n,n))
B = np.random.random(size=(n,k))
C_1 = np.zeros(shape=(n, k))
C_2 = np.zeros(shape=(n, k))
C_3 = np.zeros(shape=(n, k))
#blas level 1
B_transpose = np.transpose(B)
start_time = time.time()
for i in range(0, n):
    for j in range(0, k):
        C_1[i][j] = np.dot(A[i], B_transpose[j])
end_time = time.time()
print(f"Variant 1 took {end_time-start_time} seconds")

#blas level 2
C_transpose = np.transpose(C_1)
start_time = time.time()
for i in range(0, k):
    C_transpose[i] = np.matmul(A, B_transpose[i])
end_time = time.time()
C_2 = np.transpose(C_transpose)
print(f"Variant 2 took {end_time-start_time} seconds")
print(np.array_equal(C_1,C_2))

#blas level 3
start_time = time.time()
C_3 = A @ B
end_time = time.time()
print(f"Variant 3 took {end_time-start_time} seconds")
print(np.array_equal(C_1, C_3))

import numpy as np
from time import time

#BELOW IS SOLUTION

#Comment: from solution it is apparent that BLAS optimization only works for floats!!! not for ints.