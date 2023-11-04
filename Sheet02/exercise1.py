"""Solution of exercise 1 of exercise sheet 2"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, pprint, diff, simplify

# symbolical computations for the derivative

x = Symbol('x', real=True)
k = Symbol('k', positive=True)
kx = k * x
SoftSign = kx / (1 + abs(kx))
SoftAbs = x * SoftSign

print('##################################')
print('#    reproduce Eq. (1.21), LN    #')
print('##################################')
print('----------------------------------')
print('SoftSign_k(x) =')
pprint(SoftSign)
print('----------------------------------')
print('d(SoftSign_k(x))/dx =')
pprint(diff(SoftSign, x))
print('----------------------------------')
dSoftSign = k / (1 + abs(kx)) ** 2
print('d(SoftSign_k(x))/dx (Eq. (1.21)) =')
pprint(dSoftSign)
print('----------------------------------')
print('difference of both expressions =')
pprint(simplify(dSoftSign - diff(SoftSign, x)))
input('press <ENTER> to continue')
print('##################################')
print('#      sheet 2, exercise 1       #')
print('##################################')
print('----------------------------------')
print('SoftAbs_k(x) =')
pprint(SoftAbs)
print('----------------------------------')
print('d(SoftAbs_k(x))/dx =')
pprint(diff(SoftAbs, x))
print('----------------------------------')
dSoftAbs = kx * (2 + abs(kx)) / (1 + abs(kx)) ** 2
print('d(SoftAbs_k(x))/dx (sheet 2) =')
pprint(dSoftAbs)
print('----------------------------------')
print('difference of both expressions =')
pprint(simplify(dSoftAbs - diff(SoftAbs, x)))
input('press <ENTER> to continue')

# numerical computations / plots for symmetry and asymptotes

r = 3
xx = np.linspace(-r, r, 1001)
yy = np.abs(xx)
dd = np.sign(xx)
for kk in [1, 2, 4, 8, 16, 32]:
    kkxx = kk * xx
    softsign = kkxx / (1 + np.abs(kkxx))
    softabs = xx * softsign
    asymptote = np.abs(xx) - 1 / kk
    dsoftabs = kkxx * (2 + np.abs(kkxx)) / (1 + np.abs(kkxx)) ** 2
    plt.subplot(211)
    plt.cla()
    plt.plot(xx, softabs, label=f'SoftAbs_{kk}')
    plt.plot(xx, asymptote, label='asymptote')
    plt.plot(xx, yy, label=f'Abs')
    plt.axis([-r, r, -1, r + .1])
    plt.legend()
    plt.subplot(212)
    plt.cla()
    plt.plot(xx, dsoftabs, label=f'dSoftAbs_{kk}')
    plt.plot(xx, dd, label=f'Sign')
    plt.axis([-r, r, -1.1, 1.1])
    plt.legend()
    plt.pause(3)
