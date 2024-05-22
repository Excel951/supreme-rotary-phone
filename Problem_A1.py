import numpy as np
import pandas as pd
import PIL
import scipy
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

# Download datasets
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Split dataset
train_data, test_data = imdb['train'], imdb['test']

print(f'Length train data: {len(train_data)}')
print(f'Length test data: {len(test_data)}')

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
embedding_dim = 100
trunc_type = 'post'
oov_token = '<OOV>'

# Init Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

# Generate word index dictionary
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Generate and pad training sentences
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences=sequences, maxlen=max_length, padding='post', truncating=trunc_type)

# Generate and pad test sentences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(sequences=testing_sequences, maxlen=max_length, padding='post')

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history[f'{string}'])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, f'{string}'])
    plt.show()

# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False),
    LSTM(64, return_sequences=True),
    # tf.keras.layers.Flatten(),
    # GlobalAveragePooling1D(),
    # Dense(6, 'relu'),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1, 'sigmoid')
])

model.summary()

# Compile model
adam = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# BATCH_SIZE = 100
NUM_EPOCHS = 20

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

history_conv = model.fit(
    padded,
    training_labels_final,
    epochs=NUM_EPOCHS,
    validation_data=(testing_padded, testing_labels_final),
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# plot_graphs(history_conv, 'accuracy')
# plot_graphs(history_conv, 'mean_squared_error')
#
# # Save the model
# model.save('model_A1.h5')