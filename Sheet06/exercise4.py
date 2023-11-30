import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d

# TF/Keras uses by default float32
tf.keras.backend.set_floatx('float64')

def my_tfconv(x, f, b):

    n, h, w, c = x.shape
    fh, fw, _, m = f.shape
    y = np.zeros((n, h - fh + 1, w - fw + 1, m))

    for j in range(m):
        y[:, :, :, j] = b[j]
        for i in range(n):
            for k in range(c):
                y[i, :, :, j] += \
                  convolve2d(x[i, :, :, k],
                             f[::-1, ::-1, k, j], 'valid')

    return y

if __name__ == '__main__':

    # select sizes
    n, h, w, c = 4, 5, 5, 3
    m, fh, fw = 7, 2, 2

    # generate input
    x = np.random.randn(n, h, w, c)

    # construct TF/Keras Conv2D layer
    tf_layer = tf.keras.layers.Conv2D(m, (fh, fw))

    # compute output
    y = tf_layer(x).numpy()

    # extract weights of the layer
    f, b = tf_layer.weights

    # print shape info
    print('---------------------------')
    print('Input and output:')
    print('---------------------------')
    print(f'Input shape  = {x.shape}')
    print(f'Output shape = {y.shape}')
    print('---------------------------')
    print('Filter and bias:')
    print('---------------------------')
    print(f'Filter shape = {f.shape}')
    print(f'Bias shape   = {b.shape}')

    # replicate TF/Keras
    my_y = my_tfconv(x, f, b)

    if my_y is None:
        print('Please implement the convolutional layer in NumPy.')
    else:
        err = np.linalg.norm(y - my_y)
        if err < 1e-5:
            print(f'Success! Error = {err}')
        else:
            print(y)
            print(my_y)
