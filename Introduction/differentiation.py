import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, np.pi, 50)
y = np.cos(x)
plt.plot(x, y)
for i in range(6):
    y = np.diff(y) / np.diff(x)
    x = .5 * (x[1:] + x[:-1])
    plt.plot(x, y)

plt.legend(['function', \
            '1st derivative', \
            '2nd derivative', \
            '3rd derivative', \
            '4th derivative', \
            '5th derivative', \
            '6th derivative'])
plt.title('Numerical differentiation')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
