#%%
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
os.makedirs(models_dir, exist_ok=True)
classifier_path = os.path.join(models_dir, 'classifier.h5')
if os.path.relpath(classifier_path, models_dir) in os.listdir(models_dir):
    classifier = models.load_model(classifier_path)
else:
    classifier = get_classifier(data_features['image_shape'])
    classifier = train_classifier(classifier, dataset, batch_size=64)
    classifier.save(classifier_path)

#%% BUILD THE GENERATOR
generator_in_shape = (latent_dim + data_features['num_classes'],)
generator = get_generator(generator_in_shape)
generator.summary()

#%% BUILD THE DISCRIMINATOR
discriminator_in_shape = data_features['image_shape']
discriminator = get_discriminator(discriminator_in_shape)
discriminator.summary()

#%% BUILD GANS
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
generator_loss = get_generator_loss()
discriminator_loss = get_discriminator_loss()

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
gans.fit(dataset, epochs=10, verbose=1)

#%% SAVE THE MODELS
generator_path = os.path.join(models_dir, 'generator.h5')
discriminator_path = os.path.join(models_dir, 'discriminator.h5')
generator.save(generator_path)
discriminator.save(discriminator_path)

#%% GENERATE DATA
generate_data(latent_dim)
# %%
