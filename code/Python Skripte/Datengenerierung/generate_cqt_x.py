import utils
import numpy as np
import librosa
import pandas as pd
import sys
sys.path.insert(1, "../../nsgt/")
from nsgt import NSGT, CQ_NSGT


# Daten
files = [*["0" + str(i) for i in np.arange(1, 10, 1)], "10", "11", "12"]
files = [file + "_train" for file in files]

#files.append("01_test")

# Bestimme minimale Lämge und suche nächste Zweierpotenz
min_length = np.inf
for file in files:
    audio_data, sr = librosa.load("../../Daten/frames_data/soundfiles/" + file + "_mix_norm.wav", sr=22050)
    min_length = np.minimum(min_length, len(audio_data))
min_length = int(min_length)

i = 0
while 2 ** i < min_length:
    i = i + 1
audio_size = 2 ** i

chunk_size = int(audio_size/16)

print(chunk_size)

print("Globale Länge: " + str(audio_size))

# Parameter für CQT
fmin = librosa.note_to_hz("D1")
fmax = 22050 / 2
bins_per_octave = 5 * 12
sr = 22050
N_f = 512

# Parameter für Sliding Window
frame_length = 32
hop_length = 8

# CQT definieren
nsgt = CQ_NSGT(fmin, fmax, bins_per_octave, sr, chunk_size, matrixform=True)


offsets = ["linear", "-3", "-6"]

for offset in offsets:

    for file in files:

        print(file + "_" + offset)

        # Pfade definieren
        path_in = "../../Daten/frames_data/soundfiles/"
        path_out = "../../Daten/frames_data/spectograms/"

        # Audiodatei laden
        audio_data, sr = librosa.load(path_in + file + "_mix_norm.wav", sr=22050)
        audio_data = audio_data[:audio_size]

        i = 0
        for chunk in np.arange(0, len(audio_data), chunk_size):

            # CQT berechnen
            spec = list(nsgt.forward(audio_data[chunk : chunk + chunk_size]))

            spec = np.array([np.array([t for t in f]) for f in spec])

            print("Chunk " + str(i))

            # Aufteilen in Amplitude und Phase, nur Amplitude weiter verwenden
            spec_magn, spec_phase = utils.z2polar(spec)
            spec_magn = utils.zero_pad_spec(spec_magn, N_f - spec_magn.shape[0])

            print(spec_magn.max(), spec_magn.min())

            # Phase abspeichern
            np.save(path_out + file + "_phase_" + str(i) +".npy", spec_phase)

            # Konvertierung in dB, Normierung
            if offset == "linear":
                spec_magn_norm = utils.norm(spec_magn, 20, 0)
            else:
                spec_magn_dB = utils.to_dB(spec_magn, offset = 10 ** int(offset))
                spec_magn_norm = utils.norm(spec_magn_dB, 30, 20 * int(offset))


            # Sliding Window
            n_frames = int((spec_magn_norm.shape[1] - frame_length) / hop_length)
            frames = np.zeros(shape=(n_frames, spec_magn_norm.shape[0], frame_length))
            for frame in range(n_frames):
                frames[frame] = spec_magn_norm[:, frame * hop_length: frame * hop_length + frame_length]

            # Abspeichern
            np.save(path_out + file + "_mix_o_" + offset + "_" + str(i) + ".npy", frames)

            i = i + 1
