# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd


def remove_stopwords(sentence):
    stopwords = [
        "a", "about", "above", "after", "again", "against",
        "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "could", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers",
        "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
        "it", "it's", "its", "itself", "let's", "me", "more",
        "most", "my", "myself", "nor", "of", "on", "once",
        "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same",
        "she", "she'd", "she'll", "she's", "should",
        "so", "some", "such", "than", "that", "that's",
        "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they",
        "they'd", "they'll", "they're", "they've",
        "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was",
        "we", "we'd", "we'll", "we're", "we've",
        "were", "what", "what's", "when", "when's",
        "where", "where's", "which", "while", "who",
        "who's", "whom", "why", "why's", "with",
        "would", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves"]

    sentence = sentence.lower()

    words = sentence.split()
    no_words = [char for char in words if char not in stopwords]
    sentence = " ".join(no_words)
    return sentence


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters, or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    bbc['text'] = bbc['text'].map(remove_stopwords)
    train_datasets, validation_datasets = train_test_split(bbc, shuffle=False, train_size=training_portion)

    training_sentences, validation_sentences = train_datasets['text'], validation_datasets['text']
    training_labels, validation_labels = train_datasets['category'], validation_datasets['category']

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(sequences=training_sequences, maxlen=max_length, truncating=trunc_type,
                                    padding=padding_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(sequences=validation_sequences, maxlen=max_length, truncating=trunc_type,
                                      padding=padding_type)

    # You can also use Tokenizer to encode your label.
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc['category'])

    training_label_sequences = label_tokenizer.texts_to_sequences(training_labels)
    training_label_sequences = np.array(training_label_sequences)
    training_label_sequences = tf.convert_to_tensor(training_label_sequences)

    validation_label_sequences = label_tokenizer.texts_to_sequences(validation_labels)
    validation_label_sequences = np.array(validation_label_sequences)
    validation_label_sequences = tf.convert_to_tensor(validation_label_sequences)

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, 'relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, monitor, monitor2, target):
            super(CustomCallback, self).__init__()
            self.monitor = monitor
            self.monitor2 = monitor2
            self.target = target

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            current2 = logs.get(self.monitor2)
            if current is not None and current2 is not None:
                if current >= self.target and current2 >= self.target:
                    print(
                        f'\nEpoch {epoch + 1}: {self.monitor} and {self.monitor2} have reached {self.target}. '
                        f'Stopping training.')
                    self.model.stop_training = True

    stop_when_92 = CustomCallback('accuracy', 'val_accuracy', 0.92)

    history = model.fit(
        training_padded,
        training_label_sequences,
        epochs=500,
        validation_data=(
            validation_padded,
            validation_label_sequences
        ),
        callbacks=[stop_when_92]
    )

    loss, accuracy = model.evaluate(validation_padded, validation_label_sequences)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
