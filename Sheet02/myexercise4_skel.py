"""MNIST with self-implemented FNNs"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from tensorflow.keras.datasets import mnist
from myfeedforward_skel import FeedforwardNet

def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray],
                          tuple[np.ndarray, np.ndarray]]:
    """
    Load the MNIST dataset and preprocess the data.

    The MNIST dataset is loaded from `tensorflow` and
    then gets preprocessed. The training labels are
    one-hot encoded, and the input data is reshaped
    from shape (60000, 28, 28) to (60000, 784) and then
    transposed to (784, 60000), and (10000, 28, 28) to
    (10000, 784) to (784, 10000), respectively.

    Returns
    -------
    x_train: np.ndarray
        Training dataset, reshaped and transposed to
        shape (784, 60000)
    y_train: np.ndarray
        Training labels one-hot encoded to shape
        (10, 60000)
    x_test: np.ndarray
        Test dataset, reshaped and transposed to
        shape (784, 10000)
    y_test: np.ndarray
        Test labels of shape (10000,)
    """

    (x_train, y_train), (x_test, y_test) =\
        mnist.load_data()

    # Reshape, transpose, and scale input data
    x_train = np.reshape(x_train, (-1, 28 * 28)).T / 255.0
    x_test = np.reshape(x_test, (-1, 28 * 28)).T / 255.0

    # One-hot-encode training labels
    I = np.eye(10, 10)
    y_train = I[y_train, :]

    return (x_train, y_train), (x_test, y_test)

def load_parameters() -> tuple[tuple[ArrayLike, ArrayLike],
                               tuple[ArrayLike, ArrayLike]]:
    """
    Load the weights and biases from the file `MNIST_params.npz`

    The weights and biases are for a network with one hidden
    layer of size 100, and an output layer of size 10.

    Returns
    -------
    W1: ArrayLike
        Weight matrix for the hidden layer.
    b1: ArrayLike
        Bias for the hidden layer.
    W2: ArrayLike
        Weight matrix for the output layer.
    b2: ArrayLike
        Bias for the output layer.
    """
    
    data = np.load('Sheet02/MNIST_params.npz') 
    return (data['W1'], data['b1']), (data['W2'], data['b2'])

def get_model() -> FeedforwardNet:
    """
    Create an MNIST FNN with our implementation.

    A FNN with a hidden layer of size 100 and an
    output layer of size 10 is created. The weights
    are loaded from the file `MNIST_params.npz`.

    Returns
    -------
    FeedforwardNet
        MNIST FNN with one hidden layer of size 100 and an output layer
        of size 10.
    """

    # TODO Load weights and biases. Note that the weights
    # have to be transposed in order to use them and an
    # additional axis has to be added to the biases
    (W1, b1), (W2, b2) = load_parameters()

    # TODO Create network with one hidden layer of size 100
    # and an output layer of size 10.
    model = FeedforwardNet([784, 100, 10])

    # TODO Set the weights and the bias of the hidden layer
    model.set_weights(np.transpose(W1), 0)
    model.set_bias(np.reshape(b1, (100, 1)), 0)
    # TODO Set the weights and the bias of the output layer
    model.set_weights(np.transpose(W2), 1)
    model.set_bias(np.reshape(b2, (10, 1)), 1)
    return model

if __name__ == '__main__':

    # Create network and load weights/biases
    model = get_model()

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Evaluate the network with the MNIST test data
    m_test = model(x_test)

    # Decode predictions
    y_tilde = np.argmax(m_test, axis=0)

    # Compare the evaluated labels to the expected ones
    accuracy = np.sum(y_tilde == y_test) / 10000

    # Print the Test accuracy
    print(f'Test accuracy: {accuracy * 100}%')

    # Display some results to get a feeling
    m = 20
    for k in range(m):
        plt.imshow(x_test[:, k].reshape(28, 28), cmap='gray_r')
        plt.title(f'Label = {y_test[k]}, prediction = {y_test[k]}')
        plt.pause(1)

#visible differences include the format of the provided parameters. In our implementation, we multiply the inputs onto the weight matrix from the right
# while keras apparently uses column vectors representing the layers, and conceptually multiplies the columns of W with the input, which is a row
# vector. Additionally, the layer values are row vectors as well, s.t. the biases are row vectors as well. Other than that, mostly the same old stuff.