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
import itertools

from utils_build_models import *


def load_model(path, model_path):
    # load json and create model
    json_file = open(path + model_path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vae = model_from_json(loaded_model_json, custom_objects={'KLDivergenceLayer': KLDivergenceLayer})
    return vae


# just for Alex PC, make calculations on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#K.clear_session()

########################################Code for dynamic memory allocation########################################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
##################################################################################################################

path_models = "../../Modelle/exp_1/"


models = ["m0", "m1", "m2","m3"]
offsets = ["linear", "-3", "-6"]
instrs = ["violin", "horn", "piano"]
losses = [bc, mse]
loss_labels = ["bc", "mse"]
epochs = 1



# Lade Testdaten in Dictionaries um unnötiges Lesen und Schreiben zu verhindern
#######################################################################################################################

train_file = "op40_train"
test_file = "op40_test"


# Testdaten
y_test = {}
x_test = {}

# Trainingsdaten
y_train = {}
x_train = {}

# Inputs
for offset in offsets:
    # Trainingsdaten
    x_train[offset] = np.load("../../Daten/frames_training_data/spectograms/" + train_file + "_mix_o_" + offset + "_512_32.npy")
    x_train[offset] = x_train[offset].reshape((*x_train[offset].shape, 1))

    # Testdaten
    x_test[offset] = np.load("../../Daten/frames_test_data/spectograms/" + test_file + "_mix_o_" + offset + "_512_32.npy")
    x_test[offset] = x_test[offset].reshape((*x_test[offset].shape, 1))

print("Input-Daten geladen! \n")

# Outputs
for instr in instrs:
    y_train[instr] = {}
    y_test[instr] = {}

    for offset in offsets:
        # Trainingsdaten
        y_train[instr][offset] = np.load("../../Daten/frames_training_data/spectograms/" + train_file +
                                         "_" + instr + "_o_" + offset + "_512_32.npy")
        y_train[instr][offset] = y_train[instr][offset].reshape((*y_train[instr][offset].shape, 1))


        # Testdaten
        y_test[instr][offset] = np.load("../../Daten/frames_test_data/spectograms/"  + test_file +
                                        "_" + instr + "_o_" + offset + "_512_32.npy")
        y_test[instr][offset] = y_test[instr][offset].reshape((*y_test[instr][offset].shape, 1))

print("Output-Daten geladen! \n")
#######################################################################################################################


i = 0
steps = len(models) * len(losses) * len(offsets) * len(instrs)

for model, (loss, loss_label), offset, instr in itertools.product(models, zip(losses, loss_labels), offsets, instrs):

    print("Step " + str(i) +  "/" + str(steps))
    # Lade Modell
    if not i % int(steps/len(models)):
        vae = load_model(path_models, model)

    # compile model
    if not i % int(steps/(len(models) * len(losses))):
        vae.compile(optimizer='adam', loss=loss)
        # Setze Gewichte zurück
        initial_weights = vae.get_weights()
        new_weights = [glorot_uniform()(w.shape).eval(session=K.get_session()) for w in initial_weights]
        vae.set_weights(new_weights)

    model_name = model + "_" + loss_label + "_o_" + offset + "_" + instr

    history = History()
    save_weights_callback = keras_tools.SaveWeights(path="../../Modelle/linear_vs_dB/weights/",
                                                    string=model_name,
                                                    epochs=1)

    # train variational autoencoder
    vae.fit(x_train[offset], y_train[instr][offset],
            epochs=epochs,
            batch_size=30,
            shuffle=True,
            validation_data=(x_test[offset], y_test[instr][offset]),
            callbacks=[history, save_weights_callback]
            )

    # save history
    df = pd.DataFrame()

    for key in history.history.keys():
        df[key] = history.history[key]

    df.to_csv("../../Ergebnisse/Histories/history_" + model_name + ".csv", sep=";")

    i = i + 1
