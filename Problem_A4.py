import numpy as np
import pandas as pd
import PIL
import scipy
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

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
vocab_size = 6000
max_length = 130
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

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history[f'val_{string}'])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, f'val_{string}'])
    plt.show()

embedding_dim = 128
# kernel_size = 5
dropout_rate = 0.05

from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(32, return_sequences=True)),
    GlobalMaxPool1D(),
    Dense(20, activation='relu'),
    Dropout(0.05),
    Dense(1, activation='sigmoid')
])

# Compile model
adam = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

class CustomEarlyStoppingAccuracy83(tf.keras.callbacks.Callback):
    def __init__(self, monitor1='accuracy', monitor2='val_accuracy', target=0.83):
        super(CustomEarlyStoppingAccuracy83, self).__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        current1 = logs.get(self.monitor1)
        current2 = logs.get(self.monitor2)
        if current1 is not None and current2 is not None:
            if current1 >= self.target and current2 >= self.target:
                print(f"\nEpoch {epoch+1}: {self.monitor1} and {self.monitor2} have reached {self.target}. Stopping training.")
                self.model.stop_training = True

early_stopping = CustomEarlyStoppingAccuracy83('accuracy', 'val_accuracy', target=0.83)

BATCH_SIZE = 100
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
    callbacks=[early_stopping]
)

plot_graphs(history_conv, 'accuracy')
plot_graphs(history_conv, 'loss')

results = model.evaluate(testing_padded, testing_labels_final)

metrics_names = model.metrics_names
for name, value in zip(metrics_names, results):
    print(f'{name}: {value}')

# Save the model
model.save('model_A4.h5')