# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('./sarcasm.json', 'r') as file:
        datastore = json.load(file)

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_sentences = sentences[0:training_size]
    val_sentences = sentences[training_size:]
    print(f'Training sentences: {len(training_sentences)}')
    print(f'Val sentences: {len(val_sentences)}')

    training_labels = labels[0:training_size]
    val_labels = labels[training_size:]
    print(f'Training labels: {len(training_labels)}')
    print(f'Val labels: {len(val_labels)}')

    # Fit your tokenizer with training data
    # YOUR CODE HERE
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_seqs = tokenizer.texts_to_sequences(training_sentences)
    training_padd = pad_sequences(training_seqs, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    val_seqs = tokenizer.texts_to_sequences(val_sentences)
    val_padd = pad_sequences(val_seqs, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_labels, val_labels = np.array(training_labels), np.array(val_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, 'relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Desired accuracy and validation_accuracy > 75%
    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, monitor1, monitor2, target):
            super(CustomCallback, self).__init__()
            self.monitor1 = monitor1
            self.monitor2 = monitor2
            self.target = target

        def on_epoch_end(self, epoch, logs=None):
            current1 = logs.get(self.monitor1)
            current2 = logs.get(self.monitor2)
            if current1 is not None and current2 is not None:
                if current1 > self.target and current2 > self.target:
                    print(
                        f'\nEpoch {epoch + 1}: {self.monitor1} and {self.monitor2} have reached {self.target}. '
                        f'Stopping training.')
                    self.model.stop_training = True

    stop_when_75 = CustomCallback('accuracy', 'val_accuracy', 0.80)

    model.fit(
        training_padd,
        training_labels,
        epochs=50,
        validation_data=(val_padd, val_labels),
        callbacks=[stop_when_75]
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
