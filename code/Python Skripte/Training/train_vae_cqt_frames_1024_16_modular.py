import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tools

from keras.callbacks import History
from keras import backend as K
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, UpSampling2D, BatchNormalization, Cropping2D, MaxPooling2D
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


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


input_img = Input(shape=(1024, 16, 1))

x = Conv2D(40, (150, 1), activation ='relu', padding = "same")(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((8, 2), padding = "same")(x)
x = Conv2D(30, (20, 1), activation ='relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((4, 2), padding ='same')(x)
x = Conv2D(20, (5, 3), activation ='relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

shape = K.int_shape(x)
latent_dim = 10

x = Flatten()(x)
x = Dense(25, activation='relu')(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)


z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
z_var = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=1.0,
                                   shape=(K.shape(input_img)[0], latent_dim)))

z_eps = Multiply()([z_var, eps])
z = Add()([z_mean, z_eps])

encoder = Model(inputs = [input_img, eps], outputs = z, name='encoder')
encoder.summary()


latent_inputs = Input(shape=(latent_dim,), name='z')
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

outputs = decoder(encoder([input_img, eps]))

vae = Model(inputs=[input_img, eps], outputs=outputs)
vae.compile(optimizer='adam', loss=nll)


x_train = np.load("../../Daten/frames_training_data/pragmatic_mix_o_-6_1024_16.npy")
y_train = np.load("../../Daten/frames_training_data/pragmatic_horn_o_-6_1024_16.npy")

x_train = x_train.reshape((*x_train.shape, 1))
y_train = y_train.reshape((*y_train.shape, 1))


x_test = np.load("../../Daten/frames_training_data/moon_river_mix_o_-6_1024_16.npy")
y_test = np.load("../../Daten/frames_training_data/moon_river_horn_o_-6_1024_16.npy")

x_test = x_test.reshape((*x_test.shape, 1))
y_test= y_test.reshape((*y_test.shape, 1))


history = History()
save_weights_callback = keras_tools.SaveWeights(path ="../../Modelle/VAE/pragmatic/", string ="vae_cqt_frames_1024_16_o_-6_big_horn", epochs = 1)

# train the autoencoder
vae.fit(x_train, y_train,
        epochs=20,
        batch_size=30,
        shuffle = True,
        validation_data=(x_test, y_test),
        callbacks = [history, save_weights_callback]
        )


# save weights
vae.save_weights("../../Modelle/VAE/pragmatic/vae_cqt_frames_1024_16_o_-6_big_horn_final.h5")

df = pd.DataFrame()

for key in history.history.keys():
    df[key] = history.history[key]

df.to_csv("../../Daten/history_vae_cqt_frames_1024_16_o_-6_big_horn.csv", sep = ";")

