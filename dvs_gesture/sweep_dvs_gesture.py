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
    "width_grad": [1.0],
    "scale_grad": [0.01, 0.1, 1.0],
    # "tau_mem": [50, 100, 200, 500],  # for LIF neurons
}

# - Generate list with all combinations of parameters
configs = [dict(zip(parameters.keys(), vals)) for vals in product(*parameters.values())]

settings = {
    "num_repetitions": 3,
    "run_name": "default_sweep",
    "max_epochs": 100,  # Maximum number of training epochs.
    "method": "both",  # 'exodus', 'slayer', or 'both'
    "batch_size": 32,
    "base_channels": 2,
    "num_conv_layers": 4,
    "spike_threshold": 0.25,
    "weight_decay": 1e-2,
    "learning_rate": 1e-3,
    "bin_dt": 5000,
    "dataset_fraction": 1.0,
    "iaf": True,
    "sgd": False,
    "batchnorm": [False],
    "augmentation": True,
    "no_norm_weights": False,
    "dropout": True,
}


def make_command(kwargs, flags):
    command_string = " ".join(f"--{arg}={val}" for arg, val in kwargs.items())
    for arg, val in flags.items():
        if val:
            command_string += " --" + arg
    command_string += " --gpus=1"

    return "python train_dvs_gesture.py " + command_string


for i, cfg in enumerate(configs):
    t0 = time()
    print(f"Configuration {i+1} of {len(configs)}:")
    pprint(cfg)
    full_args = dict(cfg, **settings)
    flags = {k: v for k, v in full_args.items() if isinstance(v, bool)}
    kwargs = {k: v for k, v in full_args.items() if not isinstance(v, bool)}
    command = make_command(kwargs, flags)
    os.system(command)

    t1 = time()
    print(f"\t Took {t1-t0} s.\n")
