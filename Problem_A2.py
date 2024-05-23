# =====================================================================================
# PROBLEM A2
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import urllib.request
import zipfile

import keras.optimizers
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

def solution_A2():
    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'data/horse-or-human'
    # Augment image
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_DIR,
        batch_size=32,
        target_size=(150, 150),
        class_mode='binary'
    )

    VALIDATION_DIR = 'data/validation-horse-or-human'
    valid_datagen = ImageDataGenerator(
        rescale=1 / 255.
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=VALIDATION_DIR,
        batch_size=32,
        target_size=(150, 150),
        class_mode='binary'
    )

    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # COmpile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    class StopWhenReach90(tf.keras.callbacks.Callback):
        def __init__(self, monitor1='accuracy', monitor2='val_accuracy', target=0.90):
            super(StopWhenReach90, self).__init__()
            self.monitor1 = monitor1
            self.monitor2 = monitor2
            self.target = target

        def on_epoch_end(self, epoch, logs=None):
            current1 = logs.get(self.monitor1)
            current2 = logs.get(self.monitor2)
            if current1 is not None and current2 is not None:
                if current1 >= self.target and current2 >= self.target:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor1} and {self.monitor2} have reached {self.target}. Stopping training.")
                    self.model.stop_training = True

    custom_callback = StopWhenReach90('accuracy', 'val_accuracy', 0.87)

    # Train model
    model.fit(
        train_generator,
        epochs=20,
        validation_data=valid_generator,
        callbacks=[custom_callback]
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A2()
    model.save("model_A2.h5")
