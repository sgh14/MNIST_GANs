from tensorflow import keras
import tensorflow as tf
import numpy as np


def load_data():
    data_features = {
        'num_channels':1,
        'num_classes':10,
        'image_shape':(28, 28, 1)
    }

    # We'll use all the available examples from both the training and test sets.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    # TODO: do image preprocessing using tf.data.Dataset instead
    # Scale the pixel values to [0, 1] range, add a channel dimension to
    # the images, and one-hot encode the labels.
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1)).astype("float32") # -1 is the inferred dimension (the number of images)
    all_digits = (all_digits - 127.5) / 127.5
    all_labels = keras.utils.to_categorical(all_labels, 10)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024) #.batch(batch_size, drop_remainder=True) #.prefetch(1)
    
    return dataset, data_features