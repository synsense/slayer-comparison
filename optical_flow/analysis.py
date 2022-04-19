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
