import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# n between 2 and 99
for n in range(2, 100):
    k = np.arange(1, n+1)

    direct = n * k + (n-1) * (k-1)
    ell = np.ceil(np.log2(n+k-1))
    fft = (6 * ell + 1) * (2 ** ell)

    print(direct.shape)

    plt.clf()
    plt.plot(k, direct, 'r-', label='direct convolution')
    plt.plot(k, fft, 'b.', label='convolution via fft')
    plt.legend()
    plt.title(f'n = {n}')
    plt.pause(.1)

# n = 80 as extra plot
n = 80
k = np.arange(1, n+1)
direct = n * k + (n-1) * (k-1)
ell = np.ceil(np.log2(n+k-1))
fft = (6 * ell + 1) * (2 ** ell)

plt.clf()
plt.plot(k, direct, 'r-', label='direct convolution')
plt.plot(k, fft, 'b.', label='convolution via fft')
plt.legend()
plt.title(f'n = {n}')
plt.savefig('theory80.pdf', bbox_inches='tight')
