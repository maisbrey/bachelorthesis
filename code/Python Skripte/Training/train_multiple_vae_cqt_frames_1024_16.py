from keras.models import Model, model_from_json
from keras.callbacks import History
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Layer, Input
from keras.initializers import glorot_uniform
from keras.losses import binary_crossentropy, mean_squared_error

import keras_tools
import numpy as np
import os
import pandas as pd
import utils
import librosa

from mir_eval.separation import bss_eval_sources


def loss_1(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    return 1024 * 16 * K.mean(K.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)), axis=-1)


def loss_2(y_true, y_pred):
    """ Mean Squared Error """
    return 1024 * 16 * K.mean(K.square(K.flatten(y_true) - K.flatten(y_pred)), axis=-1)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def load_model(path, model_path):
    # load json and create model
    json_file = open(path + model_path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vae = model_from_json(loaded_model_json, custom_objects={'KLDivergenceLayer': KLDivergenceLayer})
    return vae


def frames_to_cqt(data, frame_length = 16, hop_length = 4, offset = 1e-6, max_norm = 25, min_norm = -60):
    frames = data.reshape(data.shape[0], 1024, 16)
    cqt = utils.merge_frames(frames, frame_length, hop_length)
    cqt = utils.denorm(cqt, max_norm, min_norm)
    cqt = utils.to_linear(cqt, offset = offset)
    return cqt


def zero_pad(data_1, data_2):
    # zero pad when lenghts are not the same
    if data_1.size < data_2.size:
        data_1 = np.append(data_1, np.zeros(data_2.size - data_1.size))
    elif data_2.size < data_1.size:
        data_2 = np.append(data_2, np.zeros(data_1.size - data_2.size))
    return data_1, data_2


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

# just for Alex PC, make calculations on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()

path_models = "../../Modelle/"

N_models = 12
models = ["m" + i for i in np.arange(N_models).astype(str)]
offsets = ["linear", "-3", "-6"]
instrs = ["violin", "horn", "piano"]
losses = [loss_1, loss_2]
loss_labels = ["bc", "mse"]
epochs = 100


# Lade Testdaten in Dictionaries um unnötiges Lesen und Schreiben zu verhindern
#######################################################################################################################

test_file = "op40_test"

sound_dict = {}
y_test_dict = {}
x_test_dict = {}

for instr in instrs:
    sound_dict[instr], sr = librosa.load("../../Daten/frames_test_data/soundfiles/" + test_file +
                                         "_" + instr + "_norm.wav", sr=22050)

    y_test_dict[instr] = {}

    for offset in offsets:
        y_test_dict[instr][offset] = np.load("../../Daten/frames_test_data/dB/" + test_file +
                                             "_" + instr + "_o_" + offset + "_1024_16.npy")
        y_test_dict[instr][offset] = y_test_dict[instr][offset].reshape((*y_test_dict[instr][offset].shape, 1))

for offset in offsets:
    x_test_dict[offset] = np.load("../../Daten/frames_test_data/dB/" + test_file + "_mix_o_" + offset + "_1024_16.npy")
    x_test_dict[offset] = x_test_dict[offset].reshape((*x_test_dict[offset].shape, 1))
#######################################################################################################################


for model in models:

    # load model
    vae = load_model(path_models, model)

    for loss, loss_label in zip(losses, loss_labels):

        vae.compile(optimizer='adam', loss=loss)

        # reinitialize weights
        initial_weights = vae.get_weights()
        new_weights = [glorot_uniform()(w.shape).eval(session=K.get_session()) for w in initial_weights]
        vae.set_weights(new_weights)

        for offset in offsets:

            x_train = np.load("../../Daten/frames_training_data/pragmatic_mix_o_" + offset + "_1024_16.npy")
            x_train = x_train.reshape((*x_train.shape, 1))

            x_test = np.load("../../Daten/frames_test_data/moon_river_mix_o_" + offset + "_1024_16.npy")
            x_test = x_test.reshape((*x_test.shape, 1))

            for instr in instrs:

                y_train = np.load(
                    "../../Daten/frames_training_data/pragmatic_" + instr + "_o_" + offset + "_1024_16.npy")
                y_train = y_train.reshape((*y_train.shape, 1))

                y_test = np.load("../../Daten/frames_test_data/moon_river_" + instr + "_o_" + offset + "_1024_16.npy")
                y_test = y_test.reshape((*y_test.shape, 1))


                model_name = model + "_" + loss_label + "_o_" + offset +  "_" + instr

                history = History()
                save_weights_callback = keras_tools.SaveWeights(path="../../Modelle/",
                                                                string=model_name,
                                                                epochs=1)


                # train variational autoencoder
                vae.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=30,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        callbacks=[history, save_weights_callback]
                        )

                # save history
                df = pd.DataFrame()

                for key in history.history.keys():
                    df[key] = history.history[key]

                df.to_csv("../../Ergebnisse/Histories/history_" + model_name + ".csv", sep=";")
