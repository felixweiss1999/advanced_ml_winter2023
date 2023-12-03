import tensorflow as tf
import numpy as np

def get_model():

    model = tf.keras.Sequential()

    c1_filters = 6
    c1_filter_shape = (5, 5)
    c3_filters = 16
    c3_filter_shape = (5, 5)

    model.add(tf.keras.Input(shape=(32, 32, 1)))
    model.add(tf.keras.layers.Conv2D(c1_filters, c1_filter_shape,
                                     activation='relu', name='C1'))
    model.add(tf.keras.layers.AveragePooling2D(name='S2'))
    model.add(tf.keras.layers.Conv2D(c3_filters, c3_filter_shape,
                                     activation='relu', name='C3'))
    model.add(tf.keras.layers.AveragePooling2D(name='S4'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='relu', name='C5'))
    model.add(tf.keras.layers.Dense(84, activation='relu', name='F6'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():

    model = get_model()

    x_train = np.zeros((60000, 32, 32))
    x_test = np.zeros((10000, 32, 32))

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train[:, 2:30, 2:30], y_train), (x_test[:, 2:30, 2:30],
                                        y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model.fit(x_train, y_train, batch_size=128, epochs=10)

    _, acc = model.evaluate(x_test, y_test, verbose=0)

    print(f'Accuracy: {acc * 100:5.2f}')


if __name__ == '__main__':
    main()
