from keras.callbacks import Callback
import numpy as np


class SaveWeights(Callback):

    def __init__(self, path, string, epochs = 5):
        self.path = path
        self.string = string
        self.epochs = epochs
    #def on_train_begin(self, logs={}):
        # whole model
    #    json_string = self.model.to_json()
    #    with open(self.path + self.string + ".json", "w") as json_file:
    #        json_file.write(json_string)


    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.epochs == 0:
            self.model.save_weights(self.path + self.string + "_epoch_" + str(epoch) + ".h5")

    def on_train_end(self, logs={}):
        self.model.save_weights(self.path + self.string + "_final" + ".h5")




class SaveWeightsNumpy(Callback):

    def __init__(self, path, string, epochs):
        self.path = path
        self.string = string
        self.epochs = epochs
    #def on_train_begin(self, logs={}):
        # whole model
    #    json_string = self.model.to_json()
    #    with open(self.path + self.string + ".json", "w") as json_file:
    #        json_file.write(json_string)


    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.epochs == 0:
            np.save(self.path + self.string + "_epoch_" + str(epoch) + ".npy", np.array(self.model.get_weights()))

    def on_train_end(self, logs={}):
        np.save(self.path + self.string + "_final" + ".npy", np.array(self.model.get_weights()))
        
        
def batch_generator(path, files, offset, instr, batch_size):

    while True:
        # Iteriere über Files
        for file in files:
            # Lade Daten
            x = np.load(path + file + "_mix_o_" + offset + "_512_32.npy")
            x = x.reshape((*x.shape, 1))
            y = np.load(path + file + "_" + instr + "_o_" + offset + "_512_32.npy")
            y = y.reshape((*y.shape, 1))

            # Shuffeln der Daten
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

            # Gebe Batches aus
            for batch in np.arange(0, x.shape[0], batch_size):
                yield x[batch: batch + batch_size], y[batch: batch + batch_size]

