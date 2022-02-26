import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt


def generate_data(latent_dim, generator_path='models/generator.h5', output_image_path='generated_data.png'):
  generator = models.load_model(generator_path)
  digits = list(range(10))*10
  one_hot_labels = keras.utils.to_categorical(digits, 10)
  random_latent_vectors = tf.random.normal(shape=(len(digits), latent_dim))
  random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)
  generated_images = generator(random_vector_labels)*127.5 + 127.5
  fig = plt.figure(figsize=(10,10))
  for i, img in enumerate(generated_images):
    img = img.numpy()
    plt.subplot(10, 10, i+1)
    plt.imshow(np.squeeze(img), cmap='binary')
    plt.axis('off')
    # img = keras.preprocessing.image.array_to_img(img)
    # img.save(f'digit_{i}.png')

  fig.savefig(output_image_path)