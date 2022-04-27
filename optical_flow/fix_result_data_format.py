"""
Fix result files generated in previous versions of experiments.py"""

from pathlib import Path

import pandas as pd
import numpy as np

from experiment import char_to_algo_name

result_path = Path("results")
result_files = result_path.iterdir()


def torchstring_to_array(s):
    """Remove the 'tensor' info from the strings"""
    s = s.replace("tensor(", "")
    s = s.replace(", device='cuda:0')", "")
    s = s.strip("[]")
    return np.fromstring(s, sep=",")


for j, file in enumerate(result_files):
    df = pd.read_csv(file, index_col=0)
    for col in (
        'grad_max_exodus',
        'grad_max_slayer',
        'grad_std_exodus',
        'grad_std_slayer'
    ):
        if col in df.columns:
            # Convert strings to array
            grads = np.array(
                [
                    np.fromstring(g.strip('[]'), sep=',')
                    for g in df[col]
                ]
            ).T
            # Separate algorithm from column name
            *name_components, algorithm = col.split("_")
            name = ("_").join(name_components)
            # Iterate over layers and add new columns
            for i, grad_col in enumerate(grads):
                df[f"{name}_{i}_{algorithm}"] = grad_col
            # Remove old column
            df = df.drop(columns=[col])

    if 'grad_covar' in df.columns:
        # Convert strings to array
        grads = np.array(
            [torchstring_to_array(g) for g in df['grad_covar']]
        ).T
        # Iterate over layers and add new columns
        for i, grad_col in enumerate(grads):
            df[f"grad_covar_{i}"] = grad_col
        # Remove old column
        df = df.drop(columns=['grad_covar'])

    for col in df.columns:
        # Find columns named 'grad_covar_n' (with n = 0, 1, 2,...)
        if col.startswith("grad_covar_") and len(col) == len("grad_covar_") + 1:
            # Make column name more explicit
            df.rename(columns={col: col + "_exodus_slayer"})

    if "algorithms" not in df.columns:
        # Infer used algorithms
        algorithms = ""
        for a, algo in char_to_algo_name.items():
            if f"sum_mistakes_{algo}" in df.columns:
                algorithms += a
        df["algorithms"] = algorithms

    # Store updated file
    df.to_csv(file)
    print(f"Fixed {j+1} files.", end="\r")
print("")
