import utils
import numpy as np
import os
import librosa
import pandas as pd

# choose file to create

files = [*["0" + str(i) for i in np.arange(1, 10, 1)], "11", "12"]
files = [file + "_train" for file in files]

files.append("01_test")

instrs = ["violin", "horn", "piano"]



for file in files:

    print(file)

    path = "../../Daten/frames_data/soundfiles/"

    data_dict = {}

    for instr in instrs:
        print("Lade " + instr + "...")
        data_dict[instr], sr = librosa.load(path + file + "_" + instr + ".wav", sr = 22050)

    data_dict, data_mix = utils.mix_tracks(data_dict)

    data_dict["mix"] = data_mix

    for instr in data_dict:
        librosa.output.write_wav(path + file + "_" + instr + "_norm.wav", data_dict[instr], sr)

    del data_dict
