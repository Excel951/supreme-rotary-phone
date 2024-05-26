# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf
import matplotlib.pyplot as plt


def normalize_image(images):
    images = images / 255.
    return images.reshape(images.shape[0], 28, 28, 1)


def solution_C2():
    mnist = tf.keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    train_images = normalize_image(train_images)
    test_images = normalize_image(test_images)

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, 'relu'))
    model.add(tf.keras.layers.Dense(10, 'softmax'))

    model.summary()

    # COMPILE MODEL HERE
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Desired accuracy AND validation_accuracy > 91%
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

    stop_when_91 = CustomCallback('accuracy', 'val_accuracy', 0.915)

    # TRAIN YOUR MODEL HERE
    model.fit(
        train_images,
        train_labels,
        epochs=30,
        validation_data=(test_images, test_labels),
        callbacks=[stop_when_91]
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    # model.save("model_C2.h5")
