import os

num_gpus = 2
num_epochs = 30
init_weight_path = "init_weights_smnist.pt"
dev_run = False
parameter_dict = {"tau_mem": [4, 16, 64], "scale_grad": [0.01, 0.1, 1.0]}

configs = [
    {"tau_mem": tau, "scale_grad": scale}
    for tau in parameter_dict["tau_mem"]
    for scale in parameter_dict["scale_grad"]
]


def make_command(method, tau_mem, scale_grad):
    run_name = f"smnist_tau{tau_mem}_scale{scale_grad}"
    arguments = {
        "gpus": num_gpus,
        "method": method,
        "init_weight_path": init_weight_path,
        "max_epochs": num_epochs,
        "run_name": run_name,
        "tau_mem": tau_mem,
        "scale_grad": scale_grad,
    }
    command_string = " ".join(f"--{arg}={val}" for arg, val in arguments.items())
    if dev_run:
        command_string += " --fast_dev_run"
    return "python train_smnist.py " + command_string


for cfg in configs:
    for method in ("slayer", "exodus"):
        command = make_command(**cfg, method=method)
        os.system(command)
