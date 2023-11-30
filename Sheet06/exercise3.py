import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def exercise3a(plot=False):

    model1 = tf.keras.Sequential([
        tf.keras.Input(shape=(300,), name='Input'),
        tf.keras.layers.Dense(108, name='Layer_1'),
        tf.keras.layers.Dense(64, name='Layer_2')
    ], name='DenseNetwork')

    if plot:
        tf.keras.utils.plot_model(model1, to_file='dense_model.pdf',
                                  show_shapes=True)
    model1.summary()

    model2 = tf.keras.Sequential([
        tf.keras.Input(shape=(3, 10, 10),
                       name='Input'),
        tf.keras.layers.Conv2D(3, (5, 5),
                               name='Layer_1',
                               data_format='channels_first'),
        tf.keras.layers.Conv2D(4, (3, 3),
                               name='Layer_2',
                               data_format='channels_first')
        ], name='ConvolutionalNetwork')

    if plot:
        tf.keras.utils.plot_model(model2, to_file='conv2d_model.pdf',
                                  show_shapes=True)
    model2.summary()

def exercise3b():

    # generate filter bank and bias
    # w/ different integer values
    f = np.arange(4 * 3 * 3 * 3).reshape(4, 3, 3, 3)
    c = np.arange(4)

    # generate weight matrix and bias
    W = np.tile(np.nan, (64, 108))
    b = np.zeros(64)
    for j in range(4):
        b[j*16:(j+1)*16] = c[j]
        for k in range(3):
            Tjk = np.tile(np.nan, (16, 36))
            for i in range(3):
                Tjki = np.tile(np.nan, (4, 6))
                # row zero
                Tjki[0, 0] = f[j, k, i, 2]
                Tjki[0, 1] = f[j, k, i, 1]
                Tjki[0, 2] = f[j, k, i, 0]
                # row one
                Tjki[1, 1] = f[j, k, i, 2]
                Tjki[1, 2] = f[j, k, i, 1]
                Tjki[1, 3] = f[j, k, i, 0]
                # row two
                Tjki[2, 2] = f[j, k, i, 2]
                Tjki[2, 3] = f[j, k, i, 1]
                Tjki[2, 4] = f[j, k, i, 0]
                # row three
                Tjki[3, 3] = f[j, k, i, 2]
                Tjki[3, 4] = f[j, k, i, 1]
                Tjki[3, 5] = f[j, k, i, 0]
                # blocks in Tjk
                r = (2-i)
                Tjk[0:4, r*6:(r+1)*6] = Tjki
                Tjk[4:8, (r+1)*6:(r+2)*6] = Tjki
                Tjk[8:12, (r+2)*6:(r+3)*6] = Tjki
                Tjk[12:16, (r+3)*6:(r+4)*6] = Tjki
            # blocks in W
            W[j*16:(j+1)*16, k*36:(k+1)*36] = Tjk
    plt.imshow(W, cmap='jet')
    plt.title('Weight matrix W of exercise 3b)')
    plt.savefig('weight3b.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    exercise3a()

    input('Press <Enter> to continue.')

    exercise3b()
