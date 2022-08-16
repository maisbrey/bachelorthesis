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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()

########################################Code for dynamic memory allocation########################################
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras
##################################################################################################################

path_models = "../../Modelle/exp_2/"
path_data = "../../Daten/frames_data/spectograms"

models = ["m0"]
offsets = ["linear", "-3", "-6"]
instrs = ["violin", "horn", "piano"]
losses = [bc]
loss_labels = ["bc"]

train_files = ["01_train", "02_train"]
test_file = "op40_test"

batch_size = 30

# Bestimme steps_per_epoch
n_samples = 0
for train_file in train_files:
    n_samples += np.load(path_data + train_file +"_mix_o_linear_512_32.npy",
                       mmap_mode = "r+").shape[0]

steps_per_epoch = int(n_samples / batch_size)

epochs = 10


i = 0
steps = len(models) * len(losses) * len(offsets) * len(instrs) * len(train_files)

for model, (loss, loss_label), offset, instr in itertools.product(models, zip(losses, loss_labels), offsets, instrs):

    print("Step " + str(i) + "/" + str(steps))
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



    # Testdaten
    x_test = np.load(path_data + test_file + "_mix_o_" + offset + "_512_32.npy")
    x_test = x_test.reshape((*x_test.shape, 1))

    y_test = np.load(path_data + test_file +
                                    "_" + instr + "_o_" + offset + "_512_32.npy")
    y_test = y_test.reshape((*y_test.shape, 1))


    model_name = model + "_" + loss_label + "_o_" + offset + "_" + instr

    history = History()
    save_weights_callback = keras_tools.SaveWeights(path=path_models + "weights/",
                                                    string=model_name,
                                                    epochs=1)

    # train variational autoencoder
    vae.fit_generator(keras_tools.batch_generator(path = path_data,
                                                  files=train_files,
                                                  offset = offset,
                                                  instr = instr,
                                                  batch_size = batch_size),
                      steps_per_epoch = steps_per_epoch,
                      epochs = epochs,
                      callbacks=[history, save_weights_callback],
                      validation_data = (x_test, y_test)
                      )

    # save history
    df = pd.DataFrame()

    for key in history.history.keys():
        df[key] = history.history[key]

    df.to_csv("../../Ergebnisse/exp_2/history_" + model_name + ".csv", sep=";")

    i = i + 1