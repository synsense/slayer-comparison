###
# Perform a parameter sweep by calling train_dvs_gesture.py with different configurations.
# The dict `parameters` contains the training hyper parameters to be sweeped over,
# whereas `settings` contains static settings that will be the same for all runs.
# It is possible to sweep over different settings by moving the corresponding entry
# from `settings` to `parameter` and changing the value to a list, containing the
# setings to be sweeped over.
###

from time import time
import os
from pprint import pprint
from itertools import product

parameters = {
    "rand_seed": [1, 2, 3, 4, 5],
    "scale_grad": [1.0], # [0.01, 0.1, 1.0],
    "base_channels": [8],
    "num_conv_layers": [6],
    "tau_mem": [30],  # for LIF neurons
}

# - Generate list with all combinations of parameters
configs = [dict(zip(parameters.keys(), vals)) for vals in product(*parameters.values())]

settings = {
    "max_epochs": 100,  # Maximum number of training epochs.
    "method": "exodus",  # 'exodus', 'slayer', or 'both'
    "batch_size": 20,
    "width_grad": 1.0,
    "spike_threshold": 0.25,
    "weight_decay": 1e-2,
    "learning_rate": 1e-3,
    "bin_dt": 5000,
    # "iaf": True,
    # "batchnorm": [False],
    "augmentation": True,
    "norm_weights": True,
    "dropout": True,
    "gpus": 1
}

for i, cfg in enumerate(configs):
    t0 = time()
    print(f"Configuration {i+1} of {len(configs)}:")
    pprint(cfg)
    full_args = dict(cfg, **settings)
    command_string = " ".join(f"--{arg}" if isinstance(val, bool) and val else f"--{arg}={val}" for arg, val in full_args.items())
    command_string = "python train_dvs_gesture.py " + command_string
    os.system("echo " + command_string)
    os.system(command_string)

    t1 = time()
    print(f"\t Took {t1-t0} s.\n")
