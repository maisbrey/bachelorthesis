import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#import pydot
#from keras.utils import plot_model
from utils_build_models import *


# just for Alex PC, make calculations on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#K.clear_session()

model_df = pd.DataFrame(columns = ["model", "n_filt", "conv", "pool", "dense", "latent_dim", "n_params"])

n_filts = {3: [[30, 20, 10]],
           5: [[50, 40, 30, 20, 10]]}

convs = {3: [[(31, 1), (1, 5), (5, 3)],
             [(5, 5), (5, 5), (3, 3)]],
         5: [[(31, 1), (1, 5), (15, 3), (5, 3), (3, 3)],
             [(5, 5), (5, 5), (5, 5), (5, 5), (3, 3)]]}

pools = {3: [(8, 2), (1, 4), (2, 2)],
         5: [(4, 1), (1, 4), (2, 1), (1, 2), (2, 2)]}

dense = [25]
latent_dims = [50]

N = len(n_filts.keys()) * len(n_filts[3]) * len(convs[3])

models = ["m" + i for i in np.arange(N).astype(str)]

i = 0

path = "../../Modelle/exp_1/"

# latent_dim
for latent_dim in latent_dims:
    # 3 und 5
    for n in n_filts:
        for n_filt in n_filts[n]:
            for conv in convs[n]:

                pool = pools[n]

                vae = build_model((512, 32), n_filt, conv, pool, dense, latent_dim)
                vae.summary()
                vae.compile(optimizer='adam', loss=bc)

                n_params = get_n_params([512, 32], n_filt, conv, pool, dense[0], latent_dim)

                # Abspeichern in DataFrame
                model_df.loc[i] = [models[i], n_filt, conv, pool, dense, latent_dim, n_params]

                json_string = vae.to_json()
                with open(path + models[i] + ".json", "w") as json_file:
                    json_file.write(json_string)

                i = i + 1

                del vae

model_df.to_csv(path + "model_df.csv")