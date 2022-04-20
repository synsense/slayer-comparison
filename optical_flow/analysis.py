from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

plt.ion()

result_path = Path("results")
result_files = result_path.iterdir()

results = [pd.read_csv(file, index_col=0) for file in result_files]
results = pd.concat(results, ignore_index=True)

# Get means over different parameter combinations
means = results.groupby(["grad_width", "grad_scale", "lr", "use_adam"]).mean().reset_index()

# Individual rows for exodus and slayer
means_exodus = means[["grad_width", "grad_scale", "lr", "use_adam"]].copy()
means_slayer = means[["grad_width", "grad_scale", "lr", "use_adam"]].copy()
means_exodus["algorithm"] = "exodus"
means_slayer["algorithm"] = "slayer"
means_exodus["num_successful"] = means["num_successful_exodus"]
means_slayer["num_successful"] = means["num_successful_slayer"]
means_exodus["sum_mistakes"] = means["sum_mistakes_exodus"]
means_slayer["sum_mistakes"] = means["sum_mistakes_slayer"]
for i in range(3):
    means_exodus[f"grad_max_{i}"] = means[f"grad_max_{i}_exodus"]
    means_slayer[f"grad_max_{i}"] = means[f"grad_max_{i}_slayer"]
    means_exodus[f"grad_std_{i}"] = means[f"grad_std_{i}_exodus"]
    means_slayer[f"grad_std_{i}"] = means[f"grad_std_{i}_slayer"]
data = pd.concat([means_exodus, means_slayer])

# Plotting
sns.relplot(
    data=data,
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


## -- Gradients
# Histogram about max grad distribution (layer 0)
max_grad_exodus = results[["grad_max_0_exodus"]].copy()
max_grad_exodus.rename(columns={"grad_max_0_exodus": "grad_max"})
max_grad_exodus["algorithm"] = "exodus"
max_grad_slayer = results[["grad_max_0_slayer"]].copy()
max_grad_slayer.rename(columns={"grad_max_0_slayer": "grad_max"})
max_grad_slayer["algorithm"] = "slayer"
max_grad = pd.concat([max_grad_exodus, max_grad_slayer], ignore_index=True)
sns.histplot(data=max_grad, log_scale=True, hue="algorithm", x="grad_max")

# Extreme gadients
extreme_limit = 1e7
data["grad_max_0_extreme"] = (
    pd.isna(data["grad_max_0"]) ^ (data["grad_max_0"] > extreme_limit)
)

sns.relplot(
    data=data,
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