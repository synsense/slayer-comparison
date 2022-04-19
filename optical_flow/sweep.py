from experiment import multiple_runs
from time import time
from pprint import pprint

parameters = {
    "grad_width": [.1, .5, 1, 2],
    "grad_scale": [.25, .75, 1.5],  # [.1, .5, 1, 2],
    "lr": [1e-3],  # [1e-3, 1e-2],
    "use_adam": [False],  # [False, True],
    "downsample": [1],
}

configs = [
    {
        "grad_width": grad_width,
        "grad_scale": grad_scale,
        "lr": lr,
        "use_adam": use_adam,
        "downsample": downsample,
    }
    for grad_width in parameters["grad_width"]
    for grad_scale in parameters["grad_scale"]
    for lr in parameters["lr"]
    for use_adam in parameters["use_adam"]
    for downsample in parameters["downsample"]
]

settings = {
    "load_path": "rotating_wedge_events.npy",
    "result_path": "results",
    "num_repetitions": 5,
    "num_epochs": 30,
}


for i, cfg in enumerate(configs):
    t0 = time()
    print(f"Configuration {i+1} of {len(configs)}:")
    pprint(cfg)
    fn_dict = dict(
        cfg, num_repetitions=settings["num_repetitions"], num_epochs=settings["num_epochs"]
    )
    filename = "-".join(f"{k}:{v}" for k, v in sorted(fn_dict.items()))
    multiple_runs(**dict(cfg, result_filename=filename, **settings))
    t1 = time()
    print(f"\t Took {t1-t0} s.\n")
