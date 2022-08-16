import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tools

from keras.callbacks import History
from keras import backend as K
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, UpSampling2D, BatchNormalization, Cropping2D, MaxPooling2D
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Concatenate, Activation
from keras.models import Model, Sequential

import utils


# just for Alex PC, make calculations on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()


def nll_bc(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    #return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return 1024 * 16 * K.mean(K.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)), axis=-1)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def to_linear(tensors, offset):
    tensors = [tensor * (30 - (-60)) - 60 for tensor in tensors]
    return [K.pow(tensor / 20, 10) - offset for tensor in tensors]

def to_dB(tensors, offset):
    tensors = [(tensor + 60) / (30 - (-60)) for tensor in tensors]
    return [20 * (K.log(tensor + offset) / np.log(10)) for tensor in tensors]


def masking(tensors):
    # tensors = [output_1, ..., output_N, input]
    summe = K.sum([tensor for tensor in tensors[:-1]], axis=0) + 1e-10
    masks = [tensor / summe for tensor in tensors[:-1]]
    input_img = tensors[-1]
    return [mask * input_img for mask in masks]


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mean, log_var = inputs

        kl_batch = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)

        self.add_loss(K.sum(kl_batch), inputs=inputs)

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

    n_filt.insert(0, 1)
    n_filt.pop()

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

instrs = ["violin", "horn", "piano"]

n_filts = [10, 10, 10]
convs = [(31, 1), (1, 5), (5, 3)]
pools = [(8, 2), (1, 4), (2, 2)]
dense = [25]
latent_dim = 25

offset = "linear"


# Wird mit VAEs gefüllt
vae_dict = {}

# Input Layer
input_img = Input(shape=(512, 32, 1))

# Baue so viele Encoder wie Instrumente
for instr in instrs:

    encoder, shape = build_encoder(input_img, n_filts, convs, pools, dense, latent_dim)
    #encoder.summary()

    decoder = build_decoder(n_filts, convs, pools, shape, latent_dim)
    #decoder.summary()

    outputs = decoder(encoder(input_img))

    vae_dict[instr] = Model(inputs=input_img, outputs=outputs, name = instr)


predictor_dB = Model(inputs=input_img, outputs=[vae_dict[instr](input_img) for instr in instrs], name="front")

input_masking = [Input(shape = (512, 32, 1)) for i in range(len(instrs) + 1)]
#predictions_linear = Lambda(to_linear, arguments={"offset": 10 ** int(offset)})(input_masking)
masked_linear = Lambda(masking)(input_masking)
#output_masking = Lambda(to_dB, arguments={"offset": 10 ** int(offset)})(masked_linear)


masker = Model(inputs = input_masking, outputs = masked_linear, name = "masker")
masker.summary()

masker_output = masker([*predictor_dB(input_img), input_img])

system = Model(inputs = input_img, outputs = masker_output)
system.summary()
system.compile(optimizer='adam', loss=nll_bc)

json_string = predictor_dB.to_json()
with open("../../Modelle/integrated_masking/predictor_dB.json", "w") as json_file:
    json_file.write(json_string)

json_string = system.to_json()
with open("../../Modelle/integrated_masking/system.json", "w") as json_file:
    json_file.write(json_string)

"""
# Lade Trainingsdaten
x_train = np.load("../../Daten/frames_training_data/spectograms/op40_train_mix_o_" + offset + "_512_32.npy")
x_train = x_train.reshape((*x_train.shape, 1))

y_train = {}
for instr in instrs:
    y_train[instr] = np.load("../../Daten/frames_training_data/spectograms/op40_train_" + instr + "_o_" + offset + "_512_32.npy")
    y_train[instr] = y_train[instr].reshape((*y_train[instr].shape, 1))


# Lade Testdaten
x_test = np.load("../../Daten/frames_test_data/spectograms/op40_test_mix_o_" + offset + "_512_32.npy")
x_test = x_test.reshape((*x_test.shape, 1))

y_test = {}
for instr in instrs:
    y_test[instr] = np.load("../../Daten/frames_test_data/spectograms/op40_test_" + instr + "_o_" + offset + "_512_32.npy")
    y_test[instr] = y_test[instr].reshape((*y_test[instr].shape, 1))



history = History()
save_weights_callback = callbacks.SaveWeightsNumpy(path="../../Modelle/integrated_masking/", string="system_test_weights", epochs=1)



# train variational autoencoder
system.fit(x_train, [y_train[instr] for instr in instrs],
            epochs=30,
            batch_size=30,
            shuffle=True,
            validation_data=(x_test, [y_test[instr] for instr in instrs]),
            callbacks=[history, save_weights_callback]
            )
"""