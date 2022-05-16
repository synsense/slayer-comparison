import argparse
from pathlib import Path
from tbparse import SummaryReader
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default="lightning_logs/dvs_gestures")
args = parser.parse_args()

# File handle
log_path = Path(args.logdir)
run_dirs = sorted(list(log_path.iterdir()))

# Iterate over runs
collected = []
for rd in run_dirs:
    # Load hyperparameters
    try:
        with open(rd / "version_0" / "hparams.yaml") as f:
            data = yaml.load(f, Loader=yaml.CLoader)
    except FileNotFoundError:
        print(f"No hparams.yaml found in {rd}. Skipping this directory")
        continue
    data["folder"] = rd.name

    # Load performance data
    results = SummaryReader(rd / "version_0").scalars
    valid_acc = results[results["tag"] == "valid_acc"]["value"]
    data["max_valid_acc"] = valid_acc.max()
    train_loss = results[results["tag"] == "train_loss"]["value"]
    data["min_train_loss"] = train_loss.min()
    valid_loss = results[results["tag"] == "valid_loss"]["value"]
    data["min_valid_loss"] = valid_loss.min()
    grad_norm_tags = [
        t
        for t in results["tag"].unique()
        if t.startswith("grad_2.0_norm/") and t.endswith("_v_step")
    ]
    for t in grad_norm_tags:
        t_ = (t.split("/")[1]).replace("network.", "").replace("_v_step", "")
        data["mean_gradnorm_" + t_] = results[results["tag"] == t]["value"].mean()
        data["max_gradnorm_" + t_] = results[results["tag"] == t]["value"].max()
    collected.append(data)

df = pd.DataFrame(collected)
interesting_columns = ["scale_grad", "method", "num_conv_layers", 'max_valid_acc', 'min_valid_loss', 'min_train_loss', 'folder']
interesting = df[interesting_columns]
exodus_data = interesting[interesting["method"] == "exodus"]
slayer_data = interesting[interesting["method"] == "slayer"]

dfw1 = df[df["width_grad"] == 1]
df4 = df[df["num_conv_layers"] == 4]
df8 = df[df["num_conv_layers"] == 8]

sns.scatterplot(data=df4, x="scale_grad", y="max_valid_acc", hue="method")
plt.savefig("accuracy_4lyrs.svg")
plt.figure()
sns.scatterplot(data=df8, x="scale_grad", y="max_valid_acc", hue="method")
plt.savefig("accuracy_8lyrs.svg")

## -- Grad norms
grad_cols = [c for c in dfw1.columns if c.startswith("mean_gradnorm")]
remaining_cols = set(dfw1.columns).difference(grad_cols)
grad_frames = []
for c in grad_cols:
    frame = dfw1[remaining_cols].copy()
    grad_frames.append(frame)
    frame["gradnorm"] = dfw1[c]
    frame["gradnorm_log"] = np.log(dfw1[c])
    frame["layer"] = c.replace("mean_gradnorm_", "")
    grad_frames.append(frame)
grad_norms = pd.concat(grad_frames)

gn4 = grad_norms[grad_norms["num_conv_layers"] == 4]
gn8 = grad_norms[grad_norms["num_conv_layers"] == 8]

plt.figure()
sns.scatterplot(data=gn4[gn4["method"]=="slayer"], y="gradnorm", x="layer", hue="scale_grad", s=100)
plt.yscale("log")
plt.title("slayer")
plt.savefig("gradnorms_slyr_4.svg")
plt.figure()
sns.scatterplot(data=gn8[gn8["method"]=="slayer"], y="gradnorm", x="layer", hue="scale_grad", s=100)
plt.yscale("log")
plt.title("slayer")
plt.savefig("gradnorms_slyr_8.svg")
plt.figure()
sns.scatterplot(data=gn4[gn4["method"]=="exodus"], y="gradnorm", x="layer", hue="scale_grad", s=100)
plt.yscale("log")
plt.title("exodus")
plt.savefig("gradnorms_exo_4.svg")
plt.figure()
sns.scatterplot(data=gn8[gn8["method"]=="exodus"], y="gradnorm", x="layer", hue="scale_grad", s=100)
plt.yscale("log")
plt.title("exodus")
plt.savefig("gradnorms_exo_8.svg")
