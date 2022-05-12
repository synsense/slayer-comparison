from time import time
import os
from pprint import pprint
from itertools import product

parameters = {
    "width_grad": [1.0],  # [.1, .5, 1, 2],
    "scale_grad": [.01, .1, .5, 1, 2],
    "sgd": [False],
    "num_conv_layers": [4, 8],
    "spike_threshold": [1.0],
    "weight_decay": [0.002],
}

# - Generate list with all combinations of parameters
configs = [
    dict(zip(parameters.keys(), vals))
    for vals in product(*parameters.values())
]

settings = {
    "num_repetitions": 1,
    "max_epochs": 100,
    "learning_rate": 1e-3,  # [1e-3, 1e-2],
    "method": "both",
    "batch_size": 16,
    "spatial_factor": 0.5,
    "base_channels": 2,
    "iaf": True,
    "bin_dt": 5000,
    "dataset_fraction": 1.0,
    "augmentation": False,
    "dropout": True,
    "batchnorm": False,
}


def make_command(kwargs, flags):
    command_string = " ".join(f"--{arg}={val}" for arg, val in kwargs.items())
    for arg, val in flags.items():
        if val:
            command_string += (" --" + arg)
    command_string += " --gpus=1"

    return "python train_dvs_gesture.py " + command_string

for i, cfg in enumerate(configs):
    t0 = time()
    print(f"Configuration {i+1} of {len(configs)}:")
    pprint(cfg)
    full_args = dict(cfg, **settings)
    flags = {k: v for k,v in full_args.items() if isinstance(v, bool)}
    kwargs = {k: v for k,v in full_args.items() if not isinstance(v, bool)}
    command = make_command(kwargs, flags)
    os.system(command)

    t1 = time()
    print(f"\t Took {t1-t0} s.\n")
