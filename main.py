#%%
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
import os

from data_loader import *
from classifier import *
from discriminator import get_discriminator, get_discriminator_loss
from generator import get_generator, get_generator_loss
from GANs import *
from data_generator import *

#%% LOAD DATA
dataset, data_features = load_data()
latent_dim = 128

#%% TRAIN THE CLASSIFIER IF THERE ISN'T ANY ALREADY SAVED
models_dir = 'models'
classifier_path = 'models/classifier.h5'
if os.path.relpath(classifier_path, models_dir) in os.listdir(models_dir):
    classifier = models.load_model(classifier_path)
else:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        classifier = get_classifier(data_features['image_shape'])
    
    classifier = train_classifier(classifier, dataset, batch_size=64)
    classifier.save('models/classifier.h5')

#%% BUILD THE GENERATOR
strategy = tf.distribute.MirroredStrategy()
generator_in_shape = (latent_dim + data_features['num_classes'],)
with strategy.scope():
    generator = get_generator(generator_in_shape)

generator.summary()

#%% BUILD THE DISCRIMINATOR
discriminator_in_shape = data_features['image_shape']
with strategy.scope():
    discriminator = get_discriminator(discriminator_in_shape)

discriminator.summary()

#%% BUILD GANS
with strategy.scope():
    # Instantiate the GANs model.
    gans = GANs(
        discriminator=discriminator,
        generator=generator,
        classifier=classifier,
        latent_dim=latent_dim,
        discriminator_extra_steps=2,
        generator_extra_steps=1
    )

    # Instantiate the optimizer for both networks
    generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9) 

    # Get loss functions
    generator_loss = get_generator_loss('normal')
    discriminator_loss = get_discriminator_loss('normal')

# Compile the GANs model.
gans.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
)

#%% TRAIN GANS
batch_size=64
dataset = dataset.batch(batch_size, drop_remainder=True)
gans.fit(dataset, epochs=5, verbose=1)

generator.save('models/generator.h5')
discriminator.save('models/discriminator.h5')

#%% GENERATE DATA
generate_data(latent_dim)