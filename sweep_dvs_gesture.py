from time import time
import os
from pprint import pprint
from itertools import product

parameters = {
    # "width_grad": [.5],
    "scale_grad": [1, .1],
    # "tau_mem": [50, 100, 200, 500],
    # "scale_grad": [.5],
    "sgd": [False],
    "num_conv_layers": [4],
    "spike_threshold": [0.25],
    "weight_decay": [1e-2],
    "learning_rate": [1e-3],
    "dropout": [True],
    "batchnorm": [False],
    "width_grad": [1.0],
}

# - Generate list with all combinations of parameters
configs = [dict(zip(parameters.keys(), vals)) for vals in product(*parameters.values())]

settings = {
    "num_repetitions": 3,
    "run_name": "rep_optim",
    "max_epochs": 100,
    "method": "both",
    "batch_size": 32,
    "spatial_factor": 0.5,
    "base_channels": 2,
    "iaf": True,
    "bin_dt": 5000,
    "dataset_fraction": 1.0,
    "augmentation": True,
    "norm_weights": True,
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
