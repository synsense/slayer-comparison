### --- Imports

import argparse
from typing import Tuple
from os import listdir
from pathlib import Path

import torch
from torch import nn
import numpy as np
import pandas as pd

from data import InvertDirDataset
from binary_models import SlayerModel, ExodusModel, SinabsModel


DEFAULT_PATH = "rotation_events300.npy"

char_to_model_cls = {
    "s": SlayerModel,
    "e": ExodusModel,
    "b": SinabsModel,
}

char_to_algo_name = {
    "s": "slayer",
    "e": "exodus",
    "b": "sinabs"
}

def get_models_from_str(string, kwargs_model):
    models = dict()
    for c in string:
        algo = char_to_algo_name[c]
        model_class = char_to_model_cls[c]
        models[algo] = model_class(**kwargs_model).cuda()
        models[algo].reset()
    return models

def generate_models(grad_width, grad_scale, num_timesteps, model_str: str) -> Tuple[nn.Module]:

    kwargs_model = {
        "grad_width": grad_width,
        "grad_scale": grad_scale,
        "thr": 1,
        "num_ts": num_timesteps,
    }

    # - Model generation
    models = get_models_from_str(model_str, kwargs_model)

    # - Share initial weights
    algos = list(models.keys())
    initial_params = models[algos[0]].parameter_copy
    for algo in algos[1: ]:
        models[algo].import_parameters(initial_params)

    return models


def generate_dataloader(path=DEFAULT_PATH, downsample: int = 1) -> torch.utils.data.DataLoader:
    raster = np.load(path).transpose(1, 2, 3, 0)

    num_ts = raster.shape[-1] // downsample

    # Set sample_size and step_size such that each sample corresponds to all frames of one class
    ds = InvertDirDataset(
        raster, sample_size=num_ts, step_size=num_ts, downsample=downsample
    )
    # Return dataloader
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True), num_ts


def training_step(inp, tgt, model, optim, loss_func):
    # Sum output along time axis
    output = model(inp).sum(-1).cpu()
    optim.zero_grad()
    loss = loss_func(output, tgt)
    loss.backward()
    grads = [
        p.grad.clone().detach()
        for p in model.parameters() if p.grad is not None
    ]
    optim.step()
    model.reset()

    with torch.no_grad():
        # Prediction is argmax along neuron axis
        __, prediction = torch.max(output, 1)
        mistakes = int(tgt.item() != prediction.item())

    return grads, mistakes

def training(
    models: dict,
    dataloader: torch.utils.data.DataLoader,
    lr: float = 1e-3,
    num_epochs: int = 30,
    use_adam: bool = False
):
    # - Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # - Optimizer
    optimizer_class = torch.optim.Adam if use_adam else torch.optim.SGD
    
    for name, m in models.items():
        models[name] = {"model": m, "grads": [], "mistakes": []}
        models[name]["optimizer"] = optimizer_class(m.parameters(), lr=lr)
    
    for ep in range(num_epochs):
        for inp, tgt, __ in dataloader:
            inp = inp.cuda()

            for d in models.values():
                grads, mistakes = training_step(
                    inp, tgt, d["model"], d["optimizer"], loss_func
                )
                d["grads"].append(grads)
                d["mistakes"].append(mistakes)

    # Re-arrange gradients, so that outer list is over layers and first dim. of
    # inner tensor is over time
    for d in models.values():
        d["grads"] = [
            torch.stack(g).flatten(start_dim=1)
            for g in zip(*d["grads"])
        ]

    return models

def analysis(models, result_path=None):
    results = dict()
    for algo, d in models.items():
        mistakes = np.asarray(d["mistakes"])

        # Number of mistakes
        results[f"sum_mistakes_{algo}"] = sum(mistakes)

        # Number of successful epochs
        has_mistakes = np.nonzero(mistakes)[0]
        last_mistake_idx = -1 if len(has_mistakes) == 0 else has_mistakes[-1]
        results[f"num_successful_{algo}"] = len(mistakes) - last_mistake_idx + 1

        # Largest gradient per layer
        grads = d["grads"]
        for i, g in enumerate(grads):
            results[f"grad_max_{i}_{algo}"] = torch.max(torch.abs(g)).item()
            results[f"grad_std_{i}_{algo}"] = torch.std(g).item()

    # Gradient covariances
    ## Convert keys to iterable object
    algos = list(models.keys())
    ## Iterate over all possible pairs of algorithms
    for i, algo0 in enumerate(algos):
        for algo1 in algos[i + 1: ]:
            grads0 = models[algo0]["grads"]
            grads1 = models[algo1]["grads"]

            for j, (g0, g1) in enumerate(zip(grads0, grads1)):
                # For now just look at first iteration. Not sure how to store this data otherwise
                g0 = g0[0]
                g1 = g1[0]
                enum = torch.sum(g0 * g1)
                denom = torch.sqrt(torch.sum(g0**2) * torch.sum(g1**2))
                covar = (enum / denom).item()
                results[f"grad_covar_{i}_{algo0}_{algo1}"] = covar

    if result_path is not None:
        files = listdir(result_path)
        if not files:
            file_no = 0
        else:
            last_file = sorted(files)[-1]
            file_no = int(last_file.split(".")[0]) + 1

        file_name = f"{file_no:06d}"

        np.save(Path(result_path) / file_name, results)

    return results

def single_run(
    dataloader,
    grad_width: float,
    grad_scale: float,
    num_timesteps: int,
    lr: float = 1e-3,
    num_epochs: int = 30,
    use_adam: bool = False,
    load_path=DEFAULT_PATH,
    result_path=None,
    algorithms="ebs"
):
    models = generate_models(
        grad_width, grad_scale, num_timesteps, algorithms
    )
    training_results = training(
        models, dataloader, lr, num_epochs, use_adam
    )
    return analysis(training_results, result_path=result_path)


def multiple_runs(
    grad_width: float,
    grad_scale: float,
    lr: float,
    use_adam: bool,
    num_epochs: int,
    num_repetitions: int,
    downsample: int = 1,
    load_path=DEFAULT_PATH,
    result_path="results",
    result_filename=None,
    algorithms="ebs",
):
    dataloader, num_timesteps = generate_dataloader(path=load_path, downsample=downsample)

    all_results = []

    for i in range(num_repetitions):
        print(f"Run {i + 1} of {num_repetitions}", end="\r")
        all_results.append(
            single_run(
                dataloader=dataloader,
                grad_width=grad_width,
                grad_scale=grad_scale,
                num_timesteps=num_timesteps,
                lr=lr,
                num_epochs=num_epochs,
                use_adam=use_adam,
                load_path=load_path,
                result_path=None,
                algorithms=algorithms,
            )
        )

    settings = dict(
        grad_width=grad_width,
        grad_scale=grad_scale,
        lr=lr,
        num_epochs=num_epochs,
        use_adam=use_adam,
        downsample=downsample,
        num_repetitions=num_repetitions,
        algorithms=algorithms,
    )
    df = pd.concat(
        [pd.DataFrame([dict(settings, **results)]) for results in all_results]
    )
    if result_filename is None:
        result_filename = "-".join(f"{k}:{v}" for k, v in sorted(settings.items()))

    path = Path(result_path) / f"{result_filename}.csv"
    df.to_csv(path)

    print(f"Successful. Stored results to {path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_width", "-w", type=float)
    parser.add_argument("--grad_scale", "-s", type=float)
    parser.add_argument("--lr", "-r", type=float, default=1e-3)
    parser.add_argument("--num_epochs", "-e", type=int, default=30)
    parser.add_argument("--adam", "-a", dest="use_adam", action="store_true")
    parser.add_argument("--downsample", "-d", type=int, default=1)
    parser.add_argument("--load_path", "-l", type=str, default=DEFAULT_PATH)
    parser.add_argument("--result_path", "-p", type=str, default="results")
    parser.add_argument("--result_filename", "-f", type=str, default=None)
    parser.add_argument("--num_repetitions", "-n", type=int, default=1)
    parser.add_argument("--algorithms", "-m", type=str, default="ebs")

    args = parser.parse_args()

    results = multiple_runs(
        grad_width=args.grad_width,
        grad_scale=args.grad_scale,
        lr=args.lr,
        num_epochs=args.num_epochs,
        use_adam=args.use_adam,
        downsample=args.downsample,
        num_repetitions=args.num_repetitions,
        load_path=args.load_path,
        result_path=args.result_path,
        result_filename=args.result_filename,
        algorithms=args.algorithms,
    )



