from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from experiment import char_to_algo_name

plt.ion()

result_path = Path("results")
result_files = result_path.iterdir()

results = [pd.read_csv(file, index_col=0) for file in result_files]
data = pd.concat(results, ignore_index=True)

# - Split columns into results, hyperparameters and settings
setting_cols = ["num_epochs", "downsample", "num_repetitions", "algorithmss"]
hyperparam_cols = ["grad_width", "grad_scale", "lr", "use_adam"]
result_cols = set(data.columns).difference(hyperparam_cols + setting_cols)

# Get means over different parameter combinations
means = data.groupby(hyperparam_cols + setting_cols).mean().reset_index()

# Individual rows for exodus and slayer
expanded = []

result_cols_by_algo = {
    a: [col for col in result_cols if not col.startswith("grad_covar") and algo in col]
    for a, algo in char_to_algo_name.items()
}
new_setting_cols = hyperparam_cols + setting_cols
new_setting_cols.remove("algorithms")

for a, res_cols in result_cols_by_algo.items():
    # Select rows where the algorithm has been used
    rows_with_algo = means[[a in algo for algo in means["algorithms"]]]

    new_df = rows_with_algo[new_setting_cols].copy()
    new_df["algorithm"] = char_to_algo_name[a]
    # Add relevant result columns to new algorithm
    for rc in res_cols:
        # Drop algorithm identifier from column name
        new_col = "_".join(rc.split("_")[:-1])
        new_df[new_col] = rows_with_algo[rc]
    expanded.append(new_df)

expanded_means = pd.concat(expanded, ignore_index=True)

# Plotting
sns.relplot(
    data=expanded_means.query("algorithm in ['slayer', 'exodus']"),
    x="grad_scale",
    y="grad_width",
    size="num_successful",
    hue="num_successful",
    row="lr",
    col="use_adam",
    style="algorithm",
    markers=[MarkerStyle("o", "left"), MarkerStyle("o", "right")],
    sizes=(0, 800),
    palette="flare",
)
plt.show()

# sns.relplot(
#     data=expanded_means,
#     x="grad_scale",
#     y="grad_width",
#     size="num_successful",
#     hue="num_successful",
#     row="lr",
#     col="use_adam",
#     style="algorithm",
#     markers=[MarkerStyle("o", "left"), MarkerStyle("o", "right"), MarkerStyle("o", "bottom")],
#     sizes=(0, 800),
#     palette="flare",
# )
# plt.show()


## -- Gradients
# Histogram about max grad distribution (layer 0)
max_grads = []
for algo in char_to_algo_name.values():
    new_df = data[[f"grad_max_0_{algo}"]].copy()
    new_df = new_df.rename(columns={f"grad_max_0_{algo}": "grad_max"})
    new_df["algorithm"] = algo
    max_grads.append(new_df)
max_grad = pd.concat(max_grads, ignore_index=True)
sns.kdeplot(data=max_grad, log_scale=True, hue="algorithm", x="grad_max")

# Extreme gadients
extreme_limit = 1e7
expanded_means["grad_max_0_extreme"] = (
    pd.isna(expanded_means["grad_max_0"]) ^ (expanded_means["grad_max_0"] > extreme_limit)
)
expanded_means["grad_prop_factor"] = expanded_means.eval("grad_max_0 / grad_max_2")

sns.relplot(
    data=expanded_means.query("algorithm in ['sinabs', 'exodus']"),
    x="grad_scale",
    y="grad_width",
    size="grad_max_0_extreme",
    size_order=[True, False],
    hue="algorithm",
    row="lr",
    col="use_adam",
    style="algorithm",
    markers=[MarkerStyle("o", "left"), MarkerStyle("o", "right")],
    sizes=(0, 800),
)
plt.show()

slayer_lr3_sgd = expanded_means.query("algorithm == 'slayer' & lr==0.001 & (not use_adam)")
exodus_lr3_sgd = expanded_means.query("algorithm == 'exodus' & lr==0.001 & (not use_adam)")
grad_prop_slayer = np.asarray(slayer_lr3_sgd["grad_prop_factor"]).reshape(4, 7)
grad_prop_exodus = np.asarray(exodus_lr3_sgd["grad_prop_factor"]).reshape(4, 7)
x = np.unique(exodus_lr3_sgd["grad_scale"])
y = np.unique(exodus_lr3_sgd["grad_width"])
plt.pcolor(x, y, np.log(grad_prop_slayer), vmin=2, vmax=6)
