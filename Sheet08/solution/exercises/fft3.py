import numpy as np
from scipy.linalg import dft

def fft3(x):
    n = len(x)
    if n == 1:
        y = x
    else:
        y = np.zeros(n, dtype='complex')
        m = n // 3
        omega = np.exp(-2 * np.pi * 1j / n)
        d = omega ** np.arange(m)
        om3 = np.exp(-2 * np.pi * 1j / 3)
        oc3 = om3.conj()
        z_top = fft3(x[0:n:3])
        z_mid = d * fft3(x[1:n:3])
        z_bot = d * d * fft3(x[2:n:3])
        y[0*m:1*m] = z_top +     z_mid +     z_bot
        y[1*m:2*m] = z_top + om3*z_mid + oc3*z_bot
        y[2*m:3*m] = z_top + oc3*z_mid + om3*z_bot
    return y

if __name__ == '__main__':

    k = 3
    x = np.random.randint(0, 10, 3**k)
    y_fft3 = fft3(x)
    print('Our variant =')
    print(y_fft3)
    y = np.fft.fft(x)
    print("NumPy's variant =")
    print(y)
    y_dft = dft(3**k) @ x
    print("Multiplication by SciPy's DFT matrix =")
    print(y_dft)
    print("Difference between Numpy's and our variant =",
          np.linalg.norm(y - y_fft3))
