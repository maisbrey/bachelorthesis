import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tools

from keras.callbacks import History
from keras import backend as K
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, UpSampling2D, BatchNormalization, Cropping2D, MaxPooling2D
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential


# just for Alex PC, make calculations on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()


def nll_bc(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mean, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mean) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def build_encoder(input_img, n_filt, conv, pool, dense_units, latent_dim):
    # input layer
    x = input_img

    # plenty of convolutions
    for k, i, j in zip(n_filt, conv, pool):
        x = Conv2D(k, (i[0], i[1]), activation='relu', padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((j[0], j[1]), padding="same")(x)

    # save dimensions for reconstruction
    shape = K.int_shape(x)

    # define size of latents space and flatten
    x = Flatten()(x)

    for i in dense_units:
        x = Dense(i, activation='relu')(x)
        x = BatchNormalization()(x)

    # get mean and variance
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # add KLD to loss function
    # z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])

    # sample z
    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    # build model
    encoder = Model(inputs=input_img, outputs=z, name='encoder')

    return encoder, shape

def build_decoder(n_filt, conv, pool, shape, latent_dim):
    # input layer
    latent_inputs = Input(shape=(latent_dim,), name='z')

    # reconstruct
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for k, i, j in zip(reversed(n_filt), reversed(conv), reversed(pool)):
        x = Conv2D(k, (i[0], i[1]), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((j[0], j[1]))(x)


    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding="same")(x)

    decoder = Model(latent_inputs, decoded, name='decoder')

    return decoder

n_filts = {3: [[40, 30, 20], [20, 10, 5]],
           5: [[50, 40, 30, 20, 10], [20, 15, 10, 5, 5]]}

convs = {3: [[(150, 1), (20, 1), (5, 3)],
             [(20, 1), (10, 1), (5, 3)],
             [(3, 3), (3, 3), (3, 3)]],
         5: [[(150, 1), (20, 1), (10, 2), (5, 3), (3, 3)],
             [(20, 1), (15, 1), (10, 2), (5, 3), (3, 3)],
             [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]]}

pools = {3: [(8, 2), (4, 2), (2, 2)],
         5: [(8, 2), (4, 2), (2, 2), (1, 1), (1, 1)]}

dense_units = [25]
latent_dims = [10]#, 50]

models = ["m" + i + "_eval" for i in np.arange(24).astype(str)]


i = 0

for latent_dim in latent_dims:
    for n in n_filts:
        for n_filt in n_filts[n]:
            for conv in convs[n]:

                pool = pools[n]

                input_img = Input(shape=(1024, 16, 1))

                encoder, shape = build_encoder(input_img, n_filt, conv, pool, dense_units, latent_dim)
                encoder.summary()

                decoder = build_decoder(n_filt, conv, pool, shape, latent_dim)
                decoder.summary()


                outputs = decoder(encoder(input_img))

                vae = Model(inputs=input_img, outputs=outputs)
                vae.compile(optimizer='adam', loss=nll_bc)

                # save model
                path = "../../Modelle/dB/eval/"

                json_string = vae.to_json()
                with open(path + models[i] + ".json", "w") as json_file:
                    json_file.write(json_string)

                i = i + 1








