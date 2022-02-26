import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses


def conv_block(
    x,
    filters,
    activation=None,
    kernel_size=(5, 5),
    strides=(2, 2),
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)

    return x


def get_discriminator(input_shape):
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(x, 64, layers.LeakyReLU(0.2))
    x = conv_block(x, 128, layers.LeakyReLU(0.2), use_dropout=True)
    x = conv_block(x, 256, layers.LeakyReLU(0.2), use_dropout=True)
    x = conv_block(x, 512, layers.LeakyReLU(0.2))
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    discriminator = models.Model(img_input, x, name="discriminator")

    return discriminator


def get_discriminator_loss(loss_name='normal'):
    if loss_name == 'normal':
        def discriminator_loss(batch_size, labels, logits):
            batch_size = tf.cast(batch_size, tf.float32)
            bce = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.NONE)
            loss = tf.reduce_sum(bce(labels, logits))*(1./batch_size)

            return loss
    
    elif loss_name == 'WGP':
        def discriminator_loss(real_img, fake_img): # TODO: esto está mal casi seguro
            return tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)

    return discriminator_loss
