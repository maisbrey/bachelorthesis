import numpy as np
import librosa
import bokeh
from bokeh.plotting import figure, show


def plot_spectogram(data, plot_height = 600, plot_width = 300):

    # plot stft
    plot = figure(plot_width = plot_width, plot_height = plot_height,
                  x_range = [0, data.shape[1]],
                  y_range = [0, data.shape[0]])

    plot.image(image = [data],
               x = 0, y = 0,
               dh = data.shape[0],
               dw = data.shape[1],
               palette = bokeh.palettes.Magma256)
    show(plot)


def short_sound(sound, start, len_sample):
    hann_window = 0.5 * (1 - np.cos((2 * np.pi * np.arange(len_sample))/(len_sample - 1)))
    return sound[start:start+len_sample] * hann_window



def get_stft(path, file, sr, n_fft, win_length, hop_length):

    # load data
    data, sr = librosa.load(path + file, sr = sr)

    # calculate stft
    stft = librosa.core.stft(data, n_fft = n_fft, win_length = win_length, hop_length = hop_length)

    return stft

def mel_to_stft(stft_mel, sr, n_fft, n_mels):

    # calculate mel filter
    mel_filter = librosa.filters.mel(sr, n_fft, n_mels=n_mels)

    return np.dot(np.linalg.pinv(mel_filter), stft_mel)

def get_stft_mel(path, file, sr, n_fft, win_length, hop_length, n_mels):

    # get stft
    stft = get_stft(path, file, sr, n_fft, win_length, hop_length)

    # calculate mel stft
    mel_filter = librosa.filters.mel(sr, n_fft, n_mels=n_mels)

    return np.dot(mel_filter, stft)

def get_cqt(path, file, sr, n_bins, bins_per_octave, hop_length, fmin = None):

    # load data
    data, sr = librosa.load(path + file, sr = sr)

    # calculate cqt
    cqt = librosa.core.cqt(data, sr = sr, n_bins = n_bins, bins_per_octave = bins_per_octave, hop_length = hop_length, fmin = fmin)

    return cqt


def merge_frames(frames, frame_length, hop_length):

    trans_1_rec = np.zeros(shape=(frames.shape[1], frames.shape[0] * hop_length + frame_length))

    n_frames = frames.shape[0]

    h = int(frame_length / hop_length)

    for frame in range(n_frames + h - 1):
        k = 0
        summe = np.zeros(shape=(frames[0].shape[0], hop_length))
        for i, j in zip(range(frame, frame - h, -1), range(h)):
            if i > n_frames - 1:
                continue
            elif i > -1:
                summe += frames[i][:, j * hop_length: (j + 1) * hop_length]
                k += 1
            else:
                break
        summe = summe / k

        trans_1_rec[:, hop_length * frame: hop_length * frame + hop_length] = summe

    return trans_1_rec


def z2cart(data):
    return np.abs(data), np.angle(data)

def z2polar(data):
    return np.abs(data), np.angle(data)

def cart2z(data1, data2):
    return np.add(data1, 1j * data2)

def polar2z(data1, data2):
    return np.multiply(data1, np.exp(1j * data2))

def merge(data1, data2):
    return np.stack((data1, data2), -1)

def split(data):
    return data[:,:,0], data[:,:,1]



def to_dB(data, offset = 1e-3):
    return 20 * np.log10(data + offset)

def to_linear(data, offset):
    return 10 ** (data/20) - offset


def norm(data, max, min):
    return (data - min) / (max - min)

def denorm(data, max, min):
    return (data * (max - min)) + min


def zero_pad_spec(spec, N_f):
    return np.concatenate((spec, np.zeros(shape=(N_f, spec.shape[1]))), axis=0)

def crop_spec(spec, N_f):
    return spec[:-N_f, :]



def get_pitch(string):
    start = 0
    occ = 0
    while occ < 2:
        start = string.find("_", start+1)
        occ += 1
    return int(string[start + 5:start + 8])


def filter_files(files, key, under, upper):
    return [file for file in files if file[:len(key)] == key and under < get_pitch(file) < upper]


def mix_tracks(data_dict):
    # get maximum lenght for zero padding
    max_length = 0
    for instr in data_dict:
        max_length = np.maximum(max_length, data_dict[instr].size)

    for instr in data_dict:
        # zero pad
        data_dict[instr] = np.append(data_dict[instr], np.zeros(max_length - data_dict[instr].size))

        # norm every entry on its maximum
        data_dict[instr] = data_dict[instr] / np.max(np.abs(data_dict[instr]))

    # mix data
    data_mix = np.sum([data_dict[instr] for instr in data_dict], axis=0)

    norm = np.max(np.abs(data_mix))

    for instr in data_dict:
        data_dict[instr] = data_dict[instr] / norm

    return data_dict, data_mix / norm


"""
def merge_sounds(data):

    # zero pad when lenghts are not the same
    if data_1.size < data_2.size:
        data_1 = np.append(data_1, np.zeros(data_2.size - data_1.size))
    elif data_2.size < data_1.size:
        data_2 = np.append(data_2, np.zeros(data_1.size - data_2.size))

    # norm on 1
    data_1 = data_1 / np.max(np.abs(data_1))
    data_2 = data_2 / np.max(np.abs(data_2))

    # mix data
    data = data_1 + data_2

    norm = np.max(np.abs(data))
    data_1 = data_1 / norm
    data_2 = data_2 / norm
    data = data / norm

    return data_1, data_2, data

def trans_pipeline(path, file, trans_label, repr_label, **kwargs):

    # calculate stft
    if trans_label == "linear":
        trans = get_stft(path=path, file=file, sr=kwargs["sr"], n_fft=kwargs["n_fft"], win_length=kwargs["win_length"],
                         hop_length=kwargs["hop_length"])

    # calculate_mel_stft
    elif trans_label == "mel":
        trans = get_stft_mel(path=path, file=file, sr=kwargs["sr"], n_fft=kwargs["n_fft"],
                             win_length=kwargs["win_length"], hop_length=kwargs["hop_length"], n_mels=kwargs["n_mels"])

    # calculate cqt
    elif trans_label == "cqt":
        trans = get_cqt(path=path, file=file, sr=kwargs["sr"], n_bins=kwargs["n_bins"],
                        bins_per_octave=kwargs["bins_per_octave"], hop_length=kwargs["hop_length"], fmin = kwargs["fmin"])

    else:
        print("trans_label muss aus {linear, mel, cqt} sein!")

    if repr_label == "polar":
        # split into amp and phase
        return z2polar(trans)

    elif repr_label == "cart":
        # split into real and imag part
        return z2cart(trans)

    else:
        print("repr_label muss aus {polar, cart} sein!")

def trans_pipeline(path, file, trans_label, repr_label, **kwargs):

    # calculate stft
    if trans_label == "linear":
        trans = get_stft(path = path, file = file, sr = kwargs["sr"], n_fft = kwargs["n_fft"], win_length = kwargs["win_length"], hop_length = kwargs["hop_length"])

    # calculate_mel_stft
    elif trans_label == "mel":
        trans = get_stft_mel(path = path, file = file, sr = kwargs["sr"], n_fft = kwargs["n_fft"], win_length = kwargs["win_length"], hop_length =  kwargs["hop_length"], n_mels =  kwargs["n_mels"])

    # calculate cqt
    elif trans_label == "cqt":
        trans = get_cqt(path = path, file = file, sr = kwargs["sr"], n_bins = kwargs["n_bins"], bins_per_octave = kwargs["bins_per_octave"], hop_length = kwargs["hop_length"])

    else:
        print("trans_label muss aus {linear, mel, cqt} sein!")


    # polar coordinates
    if repr_label[0] == "p":
        # split into amp and phase
        trans_1, trans_2 = z2polar(trans)
        # return both
        if repr_label[1] == "0":
            return to_dB(trans_1, kwargs["offset"]), trans_2
        # return only amp
        elif repr_label[1] == "1":
            return to_dB(trans_1, kwargs["offset"])
        #return only phase
        elif repr_label[1] == "2":
            return trans_2

    # cartesian coordinates
    elif repr_label[0] == "c":
        # split into real and imag part
        trans_1, trans_2 = z2cart(trans)
        # return both
        if repr_label[1] == "0":
            return to_dB(trans_1, kwargs["offset"]), to_dB(trans_2, kwargs["offset"])
        # return only real part
        elif repr_label[1] == "1":
            return to_dB(trans_1, kwargs["offset"])
        # return only imag part
        elif repr_label[1] == "2":
            return to_dB(trans_2, kwargs["offset"])

    print("repr_label muss die Form {p, c} + {0, 1, 2} haben! (Bspw. p0 für Polar, Amplitude und Phase)")


def norm_and_merge_pipeline(repr_label, **kwargs):

    if repr_label[1] == "0":
        # norm trans_1 and trans_2
        trans_1_norm = norm(kwargs["trans_1"], kwargs["max_1"], kwargs["min_1"])
        trans_2_norm = norm(kwargs["trans_2"], kwargs["max_2"], kwargs["min_2"])
        return merge(trans_1_norm, trans_2_norm)

    elif repr_label[1] == "1":
        # norm trans_1
        trans_1_norm = norm(kwargs["trans_1"], kwargs["max_1"], kwargs["min_1"])
        return trans_1_norm

    elif repr_label[1] == "2":
        # norm trans_2
        trans_2_norm = norm(kwargs["trans_2"], kwargs["max_2"], kwargs["min_2"])
        return trans_2_norm

    else:
        print("repr_label muss aus {0, 1, 2} sein!")


def split_and_denorm_pipeline(repr_label, **kwargs):

    if repr_label[1] == "0":
        trans_1 = denorm(kwargs["trans_1"], kwargs["max_1"], kwargs["min_1"])
        trans_2 = denorm(kwargs["trans_2"], kwargs["max_2"], kwargs["min_2"])
        if repr_label[0] == "p":
            trans_1 = to_linear(trans_1, kwargs["offset"])
        elif repr_label[0] == "c":
            trans_1 = to_linear(trans_1, kwargs["offset"])
            trans_2 = to_linear(trans_2, kwargs["offset"])
        return trans_1, trans_2

    elif repr_label[1] == "1":
        trans_1 = denorm(kwargs["trans_1"], kwargs["max_1"], kwargs["min_1"])
        return to_linear(trans_1, kwargs["offset"])

    elif repr_label[1] == "2":
        trans_2 = denorm(kwargs["trans_2"], kwargs["max_2"], kwargs["min_2"])
        if repr_label[0] == "p":
            return trans_2
        elif repr_label[0] == "cart":
            return to_linear(trans_2, kwargs["offset"])


def inverse_trans_pipeline(trans_label, repr_label, trans_1, trans_2 = None, **kwargs):

    if trans_1 == None:
        trans_1 = np.zeros(trans_2.shape)

    if trans_2 == None:
        trans_2 = np.zeros(trans_2.shape)

    if repr_label[0] == "p":
        trans = polar2z(trans_1, trans_2)
    elif repr_label[0] == "c":
        trans = cart2z(trans_1, trans_2)

    # calculate inverse stft
    if trans_label == "linear":
        return librosa.core.istft(trans, hop_length=kwargs["hop_length"])

    # calculate_inverse mel_stft
    elif trans_label == "mel":
        # convert mel scale to linear scale
        trans_linear = mel_to_stft(trans, sr = kwargs["sr"], n_fft = kwargs["n_fft"], n_mels = kwargs["n_mels"])
        return librosa.core.istft(trans, hop_length=kwargs["hop_length"])

    # calculate inverse cqt
    elif trans_label == "cqt":
        print("wird nachher gemacht!")


def output_pipeline(trans_label, repr_label, **kwargs):

    # split x_train
    rec_stft_mel_amp_dB_norm, rec_stft_mel_phase_norm = split(x_train)

    # denorm amp
    rec_stft_mel_amp_dB = denorm(rec_stft_mel_amp_dB_norm, max_amp, min_amp)

    # convert dB to linear
    rec_stft_mel_amp = to_linear(rec_stft_mel_amp_dB, offset)

    # denorm phase
    rec_stft_mel_phase = denorm(rec_stft_mel_phase_norm, max_phase, min_phase)

    # merge amp and phase
    rec_stft_mel = polar2z(rec_stft_mel_amp, rec_stft_mel_phase)

    # convert mel scale to linear scale
    rec_stft = mel_to_stft(rec_stft_mel, sr, n_fft, n_mels)

    # reconstruct sound
    return librosa.core.istft(rec_stft, hop_length=hop_length)


def input_pipeline(path, file, sr, n_fft, win_length, hop_length, n_mels, offset):

    # calculate stft
    stft = get_stft(path, file, sr, n_fft, win_length, hop_length)

    # convert linear scale to mel scale
    stft_mel = get_stft_mel(path, file, sr, n_fft, win_length, hop_length, n_mels=n_mels)

    # split into amp and phase
    stft_mel_amp, stft_mel_phase = z2polar(stft_mel)

    # norm phase
    stft_mel_phase_norm, max_min_phase = norm(stft_mel_phase)

    # convert amp to dB
    stft_mel_amp_dB = to_dB(stft_mel_amp, offset)

    # norm amp [dB]
    stft_mel_amp_dB_norm, max_min_amp = norm(stft_mel_amp_dB)

    # merge normed amp and normed phase
    x_train = merge(stft_mel_amp_dB_norm, stft_mel_phase_norm)

    return x_train, max_min_amp, max_min_phase

"""
