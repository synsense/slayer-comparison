import argparse
from pathlib import Path
from pprint import pprint
from tbparse import SummaryReader
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.ion()
sns.set(font_scale=1, style="white")


def reshape_df(df, hyperparams):
    data = pd.concat(
        {
            tag: pd.Series(
                df[df["tag"] == tag]["value"].values,
                index=df[df["tag"] == tag]["step"].values,
            ).drop_duplicates()
            for tag in df["tag"].unique()
            if tag != "hp_metric"
        },
        axis=1,
    )
    for hp, val in hyperparams.items():
        data[hp] = val
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default="lightning_logs/dvs_gestures")
args = parser.parse_args()

# File handle
log_path = Path(args.logdir)
run_dirs = sorted(list(log_path.iterdir()))

# Iterate over runs
collected_compact = []

for rd in run_dirs:
    # Load hyperparameters
    try:
        with open(rd / "version_0" / "hparams.yaml") as f:
            data = yaml.load(f, Loader=yaml.CLoader)
    except FileNotFoundError:
        print(f"No hparams.yaml found in {rd}. Skipping this directory")
        continue
    data["folder"] = rd.name

    # Load full data series
    results = SummaryReader(rd / "version_0").scalars

    ## Compact overview
    # Load performance data
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
    collected_compact.append(data)

replace = {"slayer": "SLAYER", "exodus": "EXODUS"}
compact_frame = pd.DataFrame(collected_compact).replace(replace)

dfw1 = compact_frame[compact_frame["width_grad"] == 1]
df4 = dfw1[dfw1["num_conv_layers"] == 6]
df8 = dfw1[dfw1["num_conv_layers"] == 8]

sns.scatterplot(data=df4, x="scale_grad", y="max_valid_acc", hue="method")
plt.savefig("dvs_accuracy_6lyrs.svg")
plt.figure()
sns.scatterplot(data=df8, x="scale_grad", y="max_valid_acc", hue="method")
plt.savefig("dvs_accuracy_8lyrs.svg")

## -- Grad norms
grad_cols = sorted([c for c in dfw1.columns if c.startswith("mean_gradnorm")])
grad_names = {k: f"{int(k.split('.')[1]) + 1}" for k in grad_cols if "conv" in k}
grad_names["mean_gradnorm_lin.weight"] = "FC"

remaining_cols = list(set(dfw1.columns).difference(grad_cols))
grad_frames = []
for c in grad_cols:
    frame = dfw1[remaining_cols].copy()
    grad_frames.append(frame)
    frame["Mean 2-norm"] = dfw1[c]
    frame["gradnorm_log"] = np.log(dfw1[c])
    frame["Layer"] = grad_names[c]
    grad_frames.append(frame)
grad_norms = pd.concat(grad_frames)
grad_norms = grad_norms.rename(
    columns={"method": "Algorithm", "scale_grad": "Gradient scaling"}
)


gn4 = grad_norms[grad_norms["num_conv_layers"] == 6]
gn4 = gn4[pd.isna(gn4["Mean 2-norm"]) == False]
gn8 = grad_norms[grad_norms["num_conv_layers"] == 8]

hue_kw_args = {"marker" : ["v", "X", "o",]}
for data, num_lyrs in zip([gn4, gn8], [4, 8]):
    data = data.groupby(["Algorithm", "Layer", "Gradient scaling"]).mean().reset_index()
    data = data.query("`Gradient scaling` in [0.01, 0.1, 1.0]")
    g = sns.FacetGrid(
        data,
        hue="Gradient scaling",
        col="Algorithm",
        palette="flare",
        height=3,
        aspect=0.8,
        legend_out=False,
        hue_kws=hue_kw_args,
    )
    g.map(plt.plot, "Layer", "Mean 2-norm", linestyle="-", markersize=10, color='k')
    g.set_titles("{col_name}")
    plt.yscale("log")
    axes = g.axes[0]
    axes[0].legend(
        bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.0, title="Surrogate scale"
    )
    for line in axes[0].get_lines():
        line.set_color('C0')
    for line in axes[1].get_lines():
        line.set_color("C1")
    plt.tight_layout()
    plt.savefig(f"dvs_mean_gradnorm_{num_lyrs}lyrs.svg")

## -- Convergence

hp_cols = ["scale_grad"]

# Select runs with best performance
# best_valid_acc = {4: dict(), 8: dict()}
# epoch = {4: dict(), 8: dict()}
# best_folder = {4: dict(), 8: dict()}

# sns.set(font_scale=1.5, style="white")
for num_lyrs, df in zip([6, 8], [df4, df8]):
    plt.figure(figsize=(3, 3))
    plt.title(f"Validation accuracy")  # , {num_lyrs + 1}-layer network")
    acc_list = []
    for method, plotting in zip(["EXODUS", "SLAYER"], ["-k", "--k"]):
        df_m = df[df["method"] == method]
        means = df_m.groupby(hp_cols).mean().reset_index()
        # max_index = means["max_valid_acc"].argmax()
        # max_entry = means.iloc[max_index]
        stds = df_m.groupby(hp_cols).std().reset_index()
        # Relevant hyperparams to identify runs with same set of hyperparameters
        # hp_vals = {k: max_entry[k] for k in hp_cols}
        hp_vals = {"scale_grad": 1.0}
        mean = means[means["scale_grad"] == 1.0].iloc[0]["max_valid_acc"]
        std = stds[stds["scale_grad"] == 1.0].iloc[0]["max_valid_acc"]
        query_string = " & ".join(
            [
                # Make sure that strings are marked as such
                f"({k} == '{v}')" if isinstance(v, str) else f"({k} == {v})"
                for k, v in hp_vals.items()
            ]
        )
        same_hp_entries = df_m.query(query_string)
        folders = same_hp_entries["folder"].values
        print(f"Max validation accuracy for {method}, {num_lyrs} layers:")
        print(
            # f"\t{max_entry['max_valid_acc']} +- {stds.iloc[max_index]['max_valid_acc']}"
            f"\t{mean} +- {std}"
            f" (n = {len(folders)}"
        )
        print("\twith hyperparameters:")
        pprint(hp_vals, indent=2)
        for f in folders:
            max_dir = log_path / f
            max_results = SummaryReader(max_dir / "version_0").scalars
            accs = max_results[max_results["tag"] == "valid_acc"]["value"]
            epchs = np.arange(len(accs))
            frm = pd.DataFrame(
                {"epoch": epchs, "validation accuracy": accs, "Algorithm": method}
            )
            acc_list.append(frm.reset_index())
        # best_valid_acc[num_lyrs][method] = accuracies
        # # epoch[num_lyrs][method] = max_results[max_results["tag"] == "epoch"]["value"]
        # best_folder[num_lyrs][method] = max_entry["folder"]
        # plt.plot(
        #     best_valid_acc[num_lyrs][method].values, plotting, label=method.upper()
        # )
        # max_acc = accuracies.max()
        # plt.axhline(max_acc, *plt.xlim(), ls=":", lw=0.5, color="k")

    accuracies = pd.concat(acc_list, axis=0, ignore_index=True)
    plot = sns.lineplot(
        data=accuracies,
        x="epoch",
        y="validation accuracy",
        # ci="sd",
        hue="Algorithm",
        style="Algorithm",
        dashes=((1, 0), (2, 2)),
        # color="k"
    )
    plot.axes.spines['top'].set_visible(False)
    plot.axes.spines['right'].set_visible(False)
    plt.legend(loc="best")
    plt.ylabel("Validation accuracy")
    plt.xlabel("Trained epochs")
    plt.title(" ")
    plt.tight_layout()
    plt.savefig(f"dvs_Best_acc_{num_lyrs}lyrs.svg")
