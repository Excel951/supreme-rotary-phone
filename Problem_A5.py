import numpy as np
import pandas as pd
import PIL
import scipy
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Download datasets
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Split dataset
train_data, test_data = imdb['train'], imdb['test']

# Init sentences and labels
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# Loop training examples and save sentence and label
for sentences, labels in train_data:
    training_sentences.append(sentences.numpy().decode('utf8'))
    training_labels.append(labels.numpy())

# Loop test examples and save sentence and label
for sentences, labels in test_data:
    testing_sentences.append(sentences.numpy().decode('utf8'))
    testing_labels.append(labels.numpy())

# Convert final labels into numpy array
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Param
vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_token = '<OOV>'

# Init Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

# Generate word index dictionary
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Generate and pad training sentences
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences=sequences, maxlen=max_length, truncating=trunc_type)

# Generate aand pad test sentences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(sequences=testing_sequences, maxlen=max_length)

embeddings_index = {}
with open('./glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create the embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history[f'val_{string}'])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, f'val_{string}'])
    plt.show()

# Build model
model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
adam = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()])

model.summary()

early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.2, patience=3, min_lr=1e-6)

# BATCH_SIZE = 100
NUM_EPOCHS = 50

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

validation_split = 0.2
num_val_examples = int(num_train_examples * validation_split)

history_conv = model.fit(
    padded[:-num_val_examples],
    training_labels_final[:-num_val_examples],
    epochs=NUM_EPOCHS,
    validation_data=(padded[-num_val_examples:], training_labels_final[-num_val_examples:]),
    callbacks=[early_stopping, reduce_lr]
)

plot_graphs(history_conv, 'accuracy')
plot_graphs(history_conv, 'mean_absolute_error')

results = model.evaluate(testing_padded, testing_labels_final)

metrics_names = model.metrics_names
for name, value in zip(metrics_names, results):
    print(f'{name}: {value}')

# Save the model
model.save('model_A5.h5')