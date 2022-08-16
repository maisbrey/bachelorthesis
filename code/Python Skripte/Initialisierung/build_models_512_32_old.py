import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tools

from keras.callbacks import History
from keras import backend as K
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, UpSampling2D, BatchNormalization, Cropping2D, MaxPooling2D
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Activation
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


def build_encoder(input_img, n_filt, conv, pool, dense, latent_dim):
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

    # build model
    encoder = Model(inputs=input_img, outputs=z, name="encoder")

    return encoder, shape

def build_decoder(n_filt, conv, up, shape, latent_dim):
    # input layer
    latent_inputs = Input(shape=(latent_dim,), name="z")

    # reconstruct
    x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu", name = "dense_de_1")(x)

    x = Reshape((shape[1], shape[2], shape[3]), name="reshape")(x)

    conv_de_names = ["conv_de_" + str(i) for i in range(len(conv))]
    up_de_names = ["up_de_" + str(i) for i in range(len(up))]

    #n_filt.insert(0, 1)
    #n_filt.pop()

    for k, i, j, conv_de_name, up_de_name in zip(reversed(n_filt), reversed(conv), reversed(up),
                                                   reversed(conv_de_names), reversed(up_de_names)):

        x = UpSampling2D(size = (j[0], j[1]), name=up_de_name)(x)
        x = Conv2D(k, (i[0], i[1]), padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu", name=conv_de_name)(x)

    x = Conv2D(1, (3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    decoded = Activation("sigmoid", name="output")(x)

    decoder = Model(latent_inputs, decoded, name="decoder")

    return decoder

n_filts = {3: [[30, 20, 10], [20, 20, 20], [10, 20, 30]],
           5: [[50, 40, 30, 20, 10], [30, 30, 30 , 30, 30], [10, 20, 30, 40, 50]]}

convs = {3: [[(31, 1), (1, 5), (5, 3)],
             [(5, 5), (5, 5), (3, 3)]],
         5: [[(31, 1), (1, 5), (15, 3), (5, 3), (3, 3)],
             [(5, 5), (5, 5), (5, 5), (5, 5), (3, 3)]]}

pools = {3: [(8, 2), (1, 4), (2, 2)],
         5: [(4, 1), (1, 4), (2, 1), (1, 2), (2, 2)]}

dense = [25]
latent_dims = [50]

models = ["m" + i for i in np.arange(12).astype(str)]

i = 0

#file = open("../../Modelle/exp_-1/linear_vs_dB_2_instrs/models.txt", "a")

# latent_dim
for latent_dim in latent_dims:
    # 3 und 5
    for n in n_filts:
        for n_filt in n_filts[n]:
            for conv in convs[n]:

                pool = pools[n]

                print(latent_dim, n, n_filt, conv, pool)

                input_img = Input(shape=(512, 32, 1), name = "input")

                encoder, shape = build_encoder(input_img, n_filt, conv, pool, dense, latent_dim)
                encoder.summary()

                decoder = build_decoder(n_filt.copy(), conv, pool, shape, latent_dim)
                decoder.summary()


                outputs = decoder(encoder(input_img))

                vae = Model(inputs=input_img, outputs=outputs)
                vae.summary()
                vae.compile(optimizer='adam', loss=nll_bc)

                # save model
                path = "../../Modelle/exp_-1/linear_vs_dB_2_instrs/"

                json_string = vae.to_json()
                with open(path + models[i] + ".json", "w") as json_file:
                    json_file.write(json_string)

                #file.write("\n \n"
                #      "model = " + models[i] + "\n" +
                #      "n_filt = " + str(n_filt) + "\n" +
                #      "conv: " + str(conv) + "\n"
                #      "dense_units = " + str(dense) + "\n" +
                #      "latent_dim = " + str(latent_dim) + "\n" +
                #      "\n\n######################################################################################"
                #      )

                i = i + 1

                del vae

#file.close()








