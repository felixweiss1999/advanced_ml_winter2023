# modified from:
# https://www.tensorflow.org/text/tutorials/text_generation?hl=en

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


# Elman RNN + embedding
class MySimpleRNNModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = \
          tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(rnn_units,
                                             return_sequences=True,
                                             return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None,
             return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.rnn.get_initial_state(x)
        x, states = self.rnn(x, initial_state=states,
                             training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


# output text
class OneStep(tf.keras.Model):

    def __init__(self, model, chars_from_ids,
                 ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = \
          self.model(inputs=input_ids, states=states,
                     return_state=True)
        # only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature

        # sample the output logits to generate token IDs.
        predicted_ids = \
          tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # return the characters and model state.
        return predicted_chars, states


# generate text as callback
class myCallback(Callback):

    def __init__(self, one_step_model, start_char):
        super().__init__()
        self.one_step_model = one_step_model
        self.start_char = start_char

    def on_epoch_end(self, epoch, logs=None):
        states = None
        next_char = tf.constant([self.start_char])
        result = [next_char]

        for n in range(1000):
            next_char, states = \
              self.one_step_model.generate_one_step(next_char,
                                                    states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        print('\n' + '-' * 80)
        print(result[0].numpy().decode('utf-8'))
        print('-' * 80)


def load_text(path_to_file):

    # read the file
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # length of text is the number of characters in it
    print('-'*80)
    print(f'Length of text: {len(text)} characters')

    # take a look at the first 300 characters in text
    print('-'*80)
    print('First 300 characters in text:')
    print(text[:300])

    # The unique characters in the file
    vocab = sorted(set(text))
    print('-'*80)
    print(f'list of chars = {vocab}')

    # size of vocabulary
    vocab_size = len(vocab)
    print('-'*80)
    print(f'{vocab_size} unique characters')
    print('-'*80)

    # ids <-> chars
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), num_oov_indices=0,
        mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        num_oov_indices=0, invert=True,
        mask_token=None)

    return text, vocab, vocab_size, \
      ids_from_chars, chars_from_ids

def generate_dataset(text):

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100
    examples_per_epoch = len(text) // (seq_length+1)

    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    # batch size
    BATCH_SIZE = 64

    # buffer size to shuffle the dataset
    BUFFER_SIZE = 10000

    # generate dataset
    dataset = sequences.map(split_input_target)
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    return dataset

if __name__ == '__main__':

    path_to_file = "input_frankenstein.txt"
    start_char = "F"

    # load text
    text, vocab, vocab_size, \
      ids_from_chars, chars_from_ids = load_text(path_to_file)

    # embedding dimension
    embedding_dim = 256

    # number of RNN units
    rnn_units = 1024

    model = MySimpleRNNModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        monitor='loss',
        save_best_only=True,
        save_weights_only=True)
    sample_callback = myCallback(one_step_model, start_char)

    EPOCHS = 50

    dataset = generate_dataset(text)

    history = \
      model.fit(dataset,
                epochs=EPOCHS,
                callbacks=[checkpoint_callback, sample_callback])

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss of SimpleRNN')
    plt.legend()
    plt.savefig('tf_loss.pdf', bbox_inches='tight')
    plt.show()

    tf.saved_model.save(one_step_model, 'one_step')
