import utils
import numpy as np
import librosa
import pandas as pd

files = ["op40_train", "op40_test"]
#files = ["moon_river", "pragmatic", "autumn_leaves"]

folders = ["frames_training_data/", "frames_test_data/"]
#folders = ["frames_test_data/", "frames_training_data/", "frames_training_data/"]

params = dict(sr = 22050,
              n_bins = 1024,
              bins_per_octave = 11 * 12,
              hop_length = 128,
              fmin = librosa.note_to_hz("G1")
              )

offsets = [-3, -6]
instrs = ["violin", "horn", "piano"]

for offset in offsets:

    for file, folder in zip(files, folders):
        path_in = "../../Daten/" + folder + "soundfiles/"
        path_out = "../../Daten/" + folder + "dB/"
        data_dict = {}

        for instr in instrs:

            # get transformation
            trans_1, trans_2 = utils.trans_pipeline(path = path_in, file = file + "_" + instr + "_" + "norm.wav",
                                                    trans_label = "cqt", repr_label = "polar", **params)

            # convert to dB and norm amplitude
            trans_1_dB = utils.to_dB(trans_1, offset=10 ** offset)
            trans_1_dB_norm = utils.norm(trans_1_dB, 30, 10 * offset)

            # perform sliding window
            frame_length = 16
            hop_length = 4

            n_frames = int((trans_1_dB_norm.shape[1] - frame_length) / hop_length)
            frames = np.zeros(shape=(n_frames, trans_1_dB_norm.shape[0], frame_length))

            for frame in range(n_frames):
                frames[frame] = trans_1_dB_norm[:, frame * hop_length: frame * hop_length + frame_length]

            # save amp
            np.save(path_out + file + "_" + instr + "_o_" + str(offset) + "_1024_16.npy", frames)

            print(file, instr, trans_1_dB.max())
