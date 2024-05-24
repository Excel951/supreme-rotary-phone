# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf

def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Load data fashion mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    def normalize_image(images):
        images = images/255.
        images = images.reshape(images.shape[0], 28, 28, 1)
        return images

    train_images = normalize_image(train_images)
    test_images = normalize_image(test_images)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, 'relu'))
    model.add(tf.keras.layers.Dense(10, 'softmax'))

    # COMPILE MODEL HERE
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # TRAIN YOUR MODEL HERE
    model.fit(
        train_images,
        train_labels,
        epochs=20,
        validation_data=(test_images, test_labels)
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
