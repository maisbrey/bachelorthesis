import utils
import numpy as np
import librosa
import pandas as pd
import sys
sys.path.insert(1, "../../nsgt/")
from nsgt import NSGT, CQ_NSGT

files = ["op40_train", "op40_test"]
folders = ["frames_training_data/", "frames_test_data/"]
offsets = [-3, -6]

fmin = librosa.note_to_hz("G1")
fmax = 22050 / 2
bins_per_octave = 5 * 12 + 5

N_f = 512
frame_length = 32
hop_length = 8

for offset in offsets:

    for file, folder in zip(files, folders):
        path_in = "../../Daten/" + folder + "soundfiles/"
        path_out = "../../Daten/" + folder + "dB/"
        data_dict = {}

        # load audio data
        audio_data, sr = librosa.load(path_in + file + "_mix_norm.wav", sr=22050)

        nsgt = CQ_NSGT(fmin, fmax, bins_per_octave, sr, len(audio_data), matrixform=True)

        spec = list(nsgt.forward(audio_data))
        spec = np.array([np.array([t for t in f]) for f in spec])

        spec_magn, spec_phase = utils.z2polar(spec)
        spec_magn = utils.zero_pad_spec(spec_magn, N_f - spec_magn.shape[0])

        # save phase
        np.save(path_out + file + "_phase_512.npy", spec_phase)

        spec_magn_dB = utils.to_dB(spec_magn, offset = 10 ** offset)
        spec_magn_dB_norm = utils.norm(spec_magn_dB, 30, 20 * offset)

        # perform sliding window
        n_frames = int((spec_magn_dB_norm.shape[1] - frame_length) / hop_length)
        frames = np.zeros(shape=(n_frames, spec_magn_dB_norm.shape[0], frame_length))

        for frame in range(n_frames):
            frames[frame] = spec_magn_dB_norm[:, frame * hop_length: frame * hop_length + frame_length]

        # save amp
        np.save(path_out + file + "_mix_o_" + str(offset) + "_512_32.npy", frames)

        print(file, spec_magn_dB.max())