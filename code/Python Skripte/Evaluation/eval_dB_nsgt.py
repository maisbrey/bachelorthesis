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
import sys
sys.path.insert(1, "../../nsgt/")
from nsgt import NSGT, CQ_NSGT
from utils_build_models import KLDivergenceLayer, bc, mse, sampling

from mir_eval.separation import bss_eval_sources


# just for Alex PC, make calculations on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
K.clear_session()

########################################Code for dynamic memory allocation##################################################
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras
##################################################################################################################



def load_model(path, model_path):
    # load json and create model
    json_file = open(path + model_path + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vae = model_from_json(loaded_model_json, custom_objects={'KLDivergenceLayer': KLDivergenceLayer})
    return vae

def frames_to_cqt(data, offset, frame_length = 32, hop_length = 8, max_norm = 30, min_norm = -60):
    frames = data.reshape(data.shape[0], 512, 32)
    cqt = utils.merge_frames(frames, frame_length, hop_length)
    cqt = utils.denorm(cqt, max_norm, min_norm)
    if not offset == "linear":
        cqt = utils.to_linear(cqt, offset = offset)
    return cqt

def zero_pad(data_1, data_2):
    # zero pad when lenghts are not the same
    if data_1.size < data_2.size:
        data_1 = np.append(data_1, np.zeros(data_2.size - data_1.size))
    elif data_2.size < data_1.size:
        data_2 = np.append(data_2, np.zeros(data_1.size - data_2.size))
    return data_1, data_2

def get_mse(test, pred):
    return np.sum((test - pred) ** 2)

def get_ip(test, pred):
    return np.sum(test * pred)

def get_bc(test, pred, fuzz = 1e-10):
    return np.sum((test + fuzz) * np.log((pred + fuzz) / (test + fuzz)))


# Listen für Eval Loop
models = ["m" + i for i in np.arange(12).astype(str)]
offsets = ["linear", "-3", "-6"]
instrs = ["violin", "horn", "piano"]
loss_labels = ["bc", "mse"]
epochs = np.concatenate((np.arange(0, 100, 10), [99]))

# File für Evaluation
test_file = "moon_river"

# Parameter für Sliding Window
frame_length = 32
hop_length = 8


# Dict für Modell Pfade
path_models = {"linear": "../../Modelle/linear/"}
for offset in offsets:
    path_models[offset] = "../../Modelle/dB/"

# Dicts mit Eingangs und Ausgangsdaten
y_f_ground_truth = {}
y_t_ground_truth = {}
x_test_dict = {}

# Lade Testdaten in Dictionaries um unnötiges Lesen und Schreiben zu vermeiden
for instr in instrs:
    # time domain ground truth
    y_t_ground_truth[instr], sr = librosa.load("../../Daten/frames_test_data/soundfiles/" + test_file + "_"
                                               + instr + "_norm.wav", sr=22050)

    y_f_ground_truth[instr] = {}

    # frequency domain ground truth, depending on linear/dB
    for offset in offsets:
        if offset == "linear":
            frames = np.load("../../Daten/frames_test_data/linear/" + test_file + "_" + instr + "_512_32.npy")
            y_f_ground_truth[instr][offset] = frames_to_cqt(frames, offset = "linear",
                                                            frame_length = frame_length,
                                                            hop_length = hop_length,
                                                            max_norm = 20,
                                                            min_norm = 0)
        else:
            frames = np.load("../../Daten/frames_test_data/dB/" + test_file + "_" + instr + "_o_" + offset + "_512_32.npy")
            y_f_ground_truth[instr][offset] = frames_to_cqt(frames, offset=10 ** int(offset),
                                                            frame_length=frame_length,
                                                            hop_length=hop_length,
                                                            max_norm=30,
                                                            min_norm=10 * int(offset))

# Eingangsdaten, abh. von dB/linear
for offset in offsets:
    if offset == "linear":
        x_test_dict[offset] = np.load("../../Daten/frames_test_data/linear/" + test_file + "_mix_512_32.npy")
    else:
        x_test_dict[offset] = np.load("../../Daten/frames_test_data/dB/" + test_file + "_mix_o_" + offset + "_512_32.npy")

    x_test_dict[offset] = x_test_dict[offset].reshape((*x_test_dict[offset].shape, 1))


# Linearer Original Mix
cqt_mix = frames_to_cqt(x_test_dict["linear"], offset = "linear", max_norm = 20, min_norm = 0)
# Phase
cqt_2 = np.load("../../Daten/frames_test_data/dB/" + test_file + "_phase_1024.npy")
# Counter
i = 0

error_flag = 0
# Anzahl Schritte insgesamt
steps = len(models) * len(loss_labels) * len(offsets) * len(epochs) * len(instrs)

# DataFrame für Evaluationsdaten
eval_df = pd.DataFrame(columns = ["model", "loss", "offset", "epoch", "instr",
                                  *["eval_" + offset for offset in offsets],
                                  "sdr", "sir", "sar"])

# Parameter für CQT
sr = 22050
fmin = librosa.note_to_hz('D1')
fmax = sr/2
bins_per_octave = 5 * 12

nsgt = CQ_NSGT(fmin, fmax, bins_per_octave, sr, 713847, matrixform=True)


for model, loss_label, offset, epoch in itertools.product(models, loss_labels, offsets, epochs):

    print("Step " + str(i) + "/" + str(steps))

    if not i % int(steps/len(models)):
        # load model
        vae = load_model(path_models[offset], model)

    # gets filled with cqt of different instruments
    cqt_dict = {}
    y_t_pred = {}

    # prediction for every instrument gets put in cqt_dict
    for instr in instrs:
        if offset == "linear":
            # load weights
            vae.load_weights(path_models[offset] + model + "_" + loss_label + "_" + instr + "_epoch_" + epoch + ".h5")
            # predict on instr
            cqt_dict[instr] = frames_to_cqt(vae.predict(x_test_dict[offset]), offset="linear",
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            max_norm=20,
                                            min_norm=0)
        else:
            # load weights
            vae.load_weights(path_models[offset] + model + "_" + loss_label + "_o_" + offset + "_" + instr + "_epoch_" + epoch + ".h5")
            # predict on instr
            cqt_dict[instr] = frames_to_cqt(vae.predict(x_test_dict[offset]), offset = 10 ** int(offset),
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            max_norm=30,
                                            min_norm=10 * int(offset))


    # generate masks, add phase, multiply with original mix
    cqt_sum = np.sum([cqt_dict[i] for i in cqt_dict], axis=0) + 1e-10
    for instr in instrs:
        cqt_dict[instr] = cqt_mix * cqt_dict[instr] / cqt_sum
        cqt_dict[instr] = utils.polar2z(cqt_dict[instr], cqt_2[:, :cqt_dict[instr].shape[1]])
        cqt_dict[instr] = cqt_dict[instr].reshape(cqt_dict[instr].shape[0], cqt_dict[instr].shape[1])

    # evaluation table: rows: instruments, columns: metrics
    eval = pd.DataFrame(columns = [*["eval_" + offset for offset in offsets], "sdr", "sir", "sar"],
                        index = list(cqt_dict.keys()))

    error_flag = 0

    for instr in instrs:
        # if something goes wrong
        try:
            # eval in time domain
            y_t_pred[instr] = nsgt.backward(cqt_dict[instr])

            # zero pad
            y_t_pred[instr], y_t_ground_truth[instr] = zero_pad(y_t_pred[instr], y_t_ground_truth[instr])

            # eval in frequency domain
            for offset in offsets:
                for metric in [get_mse, get_ip, get_bc]:
                    if offset == "linear":
                        eval.loc[instr, "eval_" + offset] = metric(cqt_dict[instr],
                                                                   y_f_ground_truth[instr][offset])
                    else:
                        eval.loc[instr, "eval_" + offset] = metric(utils.to_dB(cqt_dict[instr], offset = 10 ** int(offset)),
                                                                   y_f_ground_truth[instr][offset])

        except:
            error_flag = 1
            print("Error occured at: " + str([model, loss_label, offset, epoch, instr]))

    if error_flag:
        eval[:] = "error"
    else:
        eval.loc[instrs, ["sdr", "sir", "sar"]] = np.transpose(np.array(bss_eval_sources(
                                                        np.array([y_t_pred[instr] for instr in instrs]),
                                                        np.array([cqt_dict[instr] for instr in instrs]))[:3]))

    for instr in instrs:
        eval_df.loc[i] = [model, loss_label, offset, epoch, instr, *eval.loc[instr]]
        i = i + 1
        eval_df.to_csv("../../Evaluation/eval_df.csv", sep=";")