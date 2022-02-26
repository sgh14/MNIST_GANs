import tensorflow as tf
import tensorflow.keras as keras


class GANs(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        classifier,
        latent_dim,
        discriminator_extra_steps=1,
        generator_extra_steps=1
    ):
        super(GANs, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GANs, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    
    def _get_generator_inputs(self, batch_size, one_hot_labels):
        # Sample random points in the latent space and concatenate the labels.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat([random_latent_vectors, one_hot_labels], axis=1)

        return random_vector_labels


    def _get_generator_loss_and_grads(self, batch_size, g_inputs, one_hot_labels):
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(g_inputs)
            # Get the discriminator and classifier logits for fake images
            logits_from_disc = self.discriminator(generated_images)
            logits_from_clas = self.classifier(generated_images)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(logits_from_disc, logits_from_clas,
                                    tf.zeros((batch_size, 1)), one_hot_labels)

        # Get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        return g_loss, g_gradient


    def _generator_train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]
        for _ in range(self.g_steps):
            # Generate fake images from the latent vector
            g_inputs = self._get_generator_inputs(batch_size, one_hot_labels)
            g_loss, g_gradient = self._get_generator_loss_and_grads(batch_size, g_inputs, one_hot_labels)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(g_gradient, self.generator.trainable_variables)
            )

        return g_loss


    def _get_discriminator_loss_and_grads(self, images, labels):
        with tf.GradientTape() as tape:
            # Get the logits for the images
            d_logits = self.discriminator(images)
            # Calculate the discriminator loss using the fake and real image logits
            d_loss = self.d_loss_fn(labels, d_logits)
        
        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

        return d_loss, d_gradient


    def _discriminator_train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data
        batch_size = tf.shape(real_images)[0]

        for _ in range(self.d_steps):
            # Generate fake images from the latent vector
            g_inputs = self._get_generator_inputs(batch_size, one_hot_labels)
            generated_images = self.generator(g_inputs)
            combined_images = tf.concat([generated_images, real_images], axis=0)
            combined_labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
            # Get discriminator loss and gradients
            d_loss, d_gradient = self._get_discriminator_loss_and_grads(combined_images, combined_labels)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        
        return d_loss


    def train_step(self, data):
        d_loss = self._discriminator_train_step(data)
        g_loss = self._generator_train_step(data)
        
        return {"d_loss": d_loss, "g_loss": g_loss}