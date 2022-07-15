###
# Perform a parameter sweep by calling train_ssc.py with different configurations.
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
    "rand_seed": [12, 123, 1234],
    "scale_grad": [0.01, 0.1, 1.0, 2.0],
    "tau_mem": [20, 50, 200, 100000],  #100000 for IAF neurons
    "optimizer": ["adam", "sgd"]
}

# - Generate list with all combinations of parameters
configs = [dict(zip(parameters.keys(), vals)) for vals in product(*parameters.values())]

settings = {
    "batch_size": 128,
    "max_epochs": 200,
    "track_grad_norm": 2,
    "gpus": 1,
    "grad_mode": False  # Set True for getting grads of first iteration
}

def make_command(kwargs, flags):
    command_string = " ".join(f"--{arg}={val}" for arg, val in kwargs.items())
    for arg, val in flags.items():
        if val:
            command_string += " --" + arg

    return "python train_ssc.py " + command_string

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
