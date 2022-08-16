import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Lambda, UpSampling2D, MaxPooling2D, BatchNormalization
from keras.layers import Layer, Activation

from keras.models import Model


# just for Alex PC, make calculations on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()


def bc(y_true, y_pred):
    return K.sum(K.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred)), axis=-1)

def mse(y_true, y_pred):
    return K.sum(K.square(K.batch_flatten(y_true) - K.batch_flatten(y_pred)), axis=-1)

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

        kl_batch = - .5 * K.sum(1 + log_var
                                - K.square(mean)
                                - K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def build_encoder(input_img, n_filt, conv, pool, dense, latent_dim):
    """
    Erhält Konfiguration für Encoder und gibt die Schicht z (latente Repräsentation) zurück
    :param input_img: Input Layer
    :param n_filt: Array mit Anzahl der Schichten
    :param conv: Array mit Größe der Filter
    :param pool: Array mit Größe der Pooling Filter
    :param dense: Array mit Größe der Dense Layer
    :param latent_dim: Größe der latenten Repräsetation
    :return: Schicht z
    """

    # input layer
    x = input_img

    conv_en_names = ["conv_en_" + str(i) for i in range(len(conv))]
    pool_en_names = ["pool_en_" + str(i) for i in range(len(conv))]

    # plenty of convolutions
    for k, i, j, conv_en_name, pool_en_name in zip(n_filt, conv, pool, conv_en_names, pool_en_names):
        x = Conv2D(k, (i[0], i[1]), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu", name=conv_en_name)(x)
        x = MaxPooling2D(pool_size = (j[0], j[1]), padding="same", name=pool_en_name)(x)

    # save dimensions for reconstruction
    shape = K.int_shape(x)

    # define size of latents space and flatten
    x = Flatten(name="flatten")(x)

    dense_en_names = ["dense_en_" + str(i) for i in range(len(dense))]

    for i, dense_name in zip(dense, dense_en_names):
        x = Dense(i)(x)
        x = BatchNormalization()(x)
        x = Activation("relu", name=dense_name)(x)

    # get mean and variance
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # add KLD to loss function
    z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])

    # sample z
    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    return z, shape

def build_decoder(input_layer, n_filt, conv, up, shape):

    # Dense Schicht
    x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(input_layer)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu", name = "dense_de_1")(x)

    # Anordnung in ursprünglicher Form
    x = Reshape((shape[1], shape[2], shape[3]), name="reshape")(x)

    conv_de_names = ["conv_de_" + str(i) for i in range(len(conv))]
    up_de_names = ["up_de_" + str(i) for i in range(len(up))]

    for k, i, j, conv_de_name, up_de_name in zip(reversed(n_filt), reversed(conv), reversed(up),
                                                   reversed(conv_de_names), reversed(up_de_names)):

        x = UpSampling2D(size = (j[0], j[1]), name=up_de_name)(x)
        x = Conv2D(k, (i[0], i[1]), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu", name=conv_de_name)(x)

    x = Conv2D(1, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    decoded = Activation("sigmoid", name="output")(x)

    return decoded


def get_n_params(dims, n_filts, convs, pools, dense, latent_dim):
    params = 0
    n_pre = 1

    # Encoder
    for n_filt, conv, pool in zip(n_filts, convs, pools):
        # params + Filterparameter + Bias + BatchNorm
        params += n_filt * (conv[0] * conv[1] * n_pre + 1 + 4)
        n_pre = n_filt
        dims[0] = dims[0] / pool[0]
        dims[1] = dims[1] / pool[1]

    # Dense + BatchNorm
    params += dense * (dims[0] * dims[1] * n_filts[-1] + 1 + 4)
    params += 2 * latent_dim * (dense + 1)
    params += (dims[0] * dims[1] * n_filts[-1]) * (latent_dim + 1 + 4)


    # Decoder
    for n_filt, conv, pool in zip(reversed(n_filts), reversed(convs), reversed(pools)):
        # params + Filterparameter + Bias + BatchNorm
        params += n_filt * (conv[0] * conv[1] * n_pre + 1 + 4)
        n_pre = n_filt

    params += 1 * (3 * 3 * n_pre + 1 + 4)

    return int(params)


def build_model(dims, n_filt, conv, pool, dense, latent_dim):
    input_img = Input(shape=(dims[0], dims[1], 1), name="input")

    z, shape = build_encoder(input_img, n_filt, conv, pool, dense, latent_dim)

    output_decoder = build_decoder(z, n_filt, conv, pool, shape)

    return Model(inputs=input_img, outputs=output_decoder)