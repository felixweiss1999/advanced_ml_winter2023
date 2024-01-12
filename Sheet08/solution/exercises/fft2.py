import numpy as np
from scipy.linalg import dft

def fft2(x):
    n = len(x)
    if n == 1:
        y = x
    else:
        y = np.zeros(n, dtype='complex')
        m = n // 2
        omega = np.exp(-2 * np.pi * 1j / n)
        d = omega ** np.arange(m)
        z_top = fft2(x[0:n:2])
        z_bot = d * fft2(x[1:n:2])
        y[0:m] = z_top + z_bot
        y[m:n] = z_top - z_bot
    return y

if __name__ == '__main__':

    k = 5
    x = np.random.binomial(10, .5, 2**k)
    y_fft2 = fft2(x)
    print('Our variant =')
    print(y_fft2)
    y = np.fft.fft(x)
    print("NumPy's variant =")
    print(y)
    y_dft = dft(2**k) @ x
    print("Multiplication by SciPy's DFT matrix =")
    print(y_dft)
    print("Difference between Numpy's and our variant =",
          np.linalg.norm(y - y_fft2))
