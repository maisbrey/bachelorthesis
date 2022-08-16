from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, UpSampling2D, BatchNormalization, Cropping2D, MaxPooling2D, Concatenate
from keras.models import Model
from keras.callbacks import History
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import keras_tools

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

# just for Alex PC, make calculations on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


size_1 = 1024
size_2 = 16

input_img = Input(shape=(size_1, size_2, 1))

x = Conv2D(40, (150, 1), activation ='relu', padding = "same")(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((8, 2), padding = "same")(x)
x = Conv2D(30, (20, 1), activation ='relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((4, 2), padding ='same')(x)
x = Conv2D(20, (5, 3), activation ='relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)
latent_dim = 10

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(25, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, name='z')([z_mean, z_log_var])


# instantiate encoder model
encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)


x = Conv2D(20, (5, 3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(30, (20, 1), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((4, 2))(x)
x = Conv2D(40, (150, 1), activation='relu', padding = "same")(x)
x = BatchNormalization()(x)
x = UpSampling2D((8, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding = "same")(x)

# instantiate decoder model
decoder = Model(latent_inputs, decoded, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(input_img)[2])
vae = Model(input_img, outputs, name='vae')


def vae_loss(y_true, y_pred):
    reconstruction_loss = size_1 * size_2 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconstruction_loss + kl_loss

# compile model
vae.compile(optimizer='adam', loss = vae_loss)
# model summary
vae.summary()

# save model

path = "../../Modelle/"
model = "m1"

#def myprint(s):
#    summary = open(path + model + "_summary.txt", "w")
#    summary.write(s)
#    summary.close()

#vae.summary(print_fn=myprint)

# whole model
json_string = vae.to_json()
with open(path + model + ".json", "w") as json_file:
    json_file.write(json_string)




