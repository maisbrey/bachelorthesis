from keras.models import Model, model_from_json
from keras import backend as K

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

def frames_to_cqt(data, offset, frame_length = 32, hop_length = 8, max_norm = 30, min_norm = -60):
    frames = data.reshape(data.shape[0], 512, 32)
    cqt = utils.merge_frames(frames, frame_length, hop_length)
    cqt = utils.denorm(cqt, max_norm, min_norm)
    if not offset == "linear":
        cqt = utils.to_linear(cqt, offset = offset)
    return cqt


def load_model(path, model):
    # load json and create model
    json_file = open(path + model + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vae = model_from_json(loaded_model_json, custom_objects={'KLDivergenceLayer': KLDivergenceLayer})
    # evaluate loaded model on test data
    vae.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return vae


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

def metric_f(y_test, y_pred):
    return - np.sum(y_test * np.log(y_pred + epsilon))


instrs = ["violin", "horn"]
models = ["m" + i for i in np.arange(4).astype(str)]
losses = ["bc", "mse"]
offsets = ["linear", "-3", "-6"]

################################ Bestimme bestes Modell #########################################

best_model = pd.DataFrame(columns=["model", "loss", "offset", "instr", "epoch"])

i = 0
for model, loss, offset, instr in itertools.product(models, losses, offsets, instrs):
    history = pd.read_csv("../Ergebnisse/exp_-1/Histories_2_instrs/" + "history_" +
                          model + "_" + loss + "_o_" + offset + "_" + instr + ".csv", sep=";")

    best_model.loc[i] = [model, loss, offset, instr, history["val_loss"].idxmin()]
    i = i + 1

################################## Lade Ground Truth #########################################

frame_length = 32
hop_length = 8

test_file = "moon_river"

y_t_ground_truth = {}
y_f_ground_truth = {}

# Lade Ground Truth
for instr in instrs:
    # Zeitbereich Ground Truth
    y_t_ground_truth[instr], sr = librosa.load("../Daten/frames_test_data/soundfiles/" + test_file + "_"
                                               + instr + "_norm.wav", sr=22050)

    # Frequenzbereich Ground_Truth
    y_f_ground_truth[instr] = frames_to_cqt(np.load("../Daten/frames_test_data/spectograms/" + test_file +
                                                    "_" + instr + "_o_linear_512_32.npy"),
                                            offset="linear",
                                            max_norm=20, min_norm=0)
print("Daten geladen!")

###################################### Definiere CQT, lade Original Mix ################################################

# Parameter für CQT
sr = 22050
fmin = librosa.note_to_hz('D1')
fmax = sr / 2
bins_per_octave = 5 * 12
nsgt = CQ_NSGT(fmin, fmax, bins_per_octave, sr, y_t_ground_truth["violin"].size, matrixform=True)


x_test = frames_to_cqt(np.load("../Daten/frames_test_data/spectograms/" + test_file + "_mix_o_linear_512_32.npy"),
                       offset="linear",
                       max_norm=20,
                       min_norm=0)

# Linearer Original Mix
cqt_mix = frames_to_cqt(x_test, offset = "linear", max_norm = 20, min_norm = 0)


# Lade Phase
cqt_2 = np.load("../Daten/frames_test_data/spectograms/" + test_file + "_phase_512.npy")

# Pfad zu Gewichten
path = "../Modelle/exp_-1/linear_vs_dB_2_instrs/"

#######################################Iteriere über alle Modelle#######################################################

# DataFrame für Evaluationsdaten
eval_df = pd.DataFrame(columns = ["model", "loss", "offset", "epoch", "instr", "eval", "sdr", "sir", "sar"])

for i in np.arange(0, len(best_model["instr"]), len(instrs)):

    (model, loss, offset) = best_model.loc[i, ["model", "loss", "offset"]]

    # Lade Modell
    vae = load_model(path, model)

    # Lade Testdaten
    x_test = np.load("../Daten/frames_test_data/spectograms/moon_river_mix_o_" + offset + "_512_32.npy")
    x_test = x_test.reshape((*x_test.shape, 1))

    y_test = {}
    for instr in ["violin", "horn"]:
        y_test[instr] = np.load("../Daten/frames_test_data/spectograms/moon_river_" +
                                instr + "_o_" + offset + "_512_32.npy")
        y_test[instr] = y_test[instr].reshape((*y_test[instr].shape, 1))


    cqt_dict = {}

    # Treffe Vorhersagen
    for instr, k in zip(instrs, range(len(instrs))):
        epoch = best_model.loc[i + k, "epoch"]
        vae.load_weights(path + "weights/" + model + "_" + loss + "_o_" + offset + "_" + instr + "_epoch_" +
                         str(epoch) + ".h5")

        if offset == "linear":
            max_norm, min_norm = (20, 0)
        else:
            max_norm, min_norm = (30, 20 * int(offset))

        # Frames werden wieder in ein Stück angeordnet
        cqt_dict[instr] = frames_to_cqt(vae.predict(x_test),
                                        offset=offset,
                                        frame_length=frame_length,
                                        hop_length=hop_length,
                                        max_norm=max_norm,
                                        min_norm=min_norm)

        # Maskieren mit den Ergebnissen anderer Modelle
        cqt_sum = np.sum([cqt_dict[i] for i in cqt_dict], axis=0) + 1e-10
        for instr in instrs:
            cqt_dict[instr] = cqt_mix * cqt_dict[instr] / cqt_sum
            cqt_dict[instr] = cqt_dict[instr].reshape(cqt_dict[instr].shape[0], cqt_dict[instr].shape[1])

    # Evaluation
    eval_t_f = pd.DataFrame(columns=[*["eval_" + offset for offset in offsets], # Verschiedene Frequenzen
                                     "sdr", "sir", "sar"],
                            index=list(cqt_dict.keys()))

    y_t_pred = {}

    for instr in instrs:
        # Fängt Fehler auf
        try:

            # Evaluation im Frequenzbereich
            # Zu diesem Zeitpunkt sind bereits alle Offsets linear
            eval_t_f.loc[instr, *["eval_" + offset for offset in offsets]] = metric_f(cqt_dict[instr], y_f_ground_truth[instr][offset])

            cqt_dict[instr] = utils.polar2z(cqt_dict[instr], cqt_2[:, :cqt_dict[instr].shape[1]])

            # Rückstransformation in Zeitbereich
            y_t_pred[instr] = nsgt.backward(cqt_dict[instr])

            # Zero-Padding
            y_t_pred[instr], y_t_ground_truth[instr] = zero_pad(y_t_pred[instr], y_t_ground_truth[instr])



        except:
            error_flag = 1
            print("Error occured at: " + str([model, loss, offset, epoch, instr]))

    if error_flag:
        eval_t_f[:] = "error"
        error_flag = 0
    else:
        eval_t_f.loc[instrs, ["sdr", "sir", "sar"]] = np.transpose(np.array(bss_eval_sources(
            np.array([y_t_pred[instr] for instr in instrs]),
            np.array([cqt_dict[instr] for instr in instrs]))[:3]))

    for instr in instrs:
        eval_df.loc[i] = [model, loss, offset, epoch, instr, *eval_t_f.loc[instr]]
        i = i + 1
        eval_df.to_csv("../../Evaluation/eval_df.csv", sep=";")