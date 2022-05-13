from pathlib import Path
from tbparse import SummaryReader
import yaml
import pandas as pd

# File handle
log_path = Path("lightning_logs/dvs/both")
run_dirs = sorted(list(log_path.iterdir()))

# Iterate over runs
collected = []
for rd in run_dirs:
    # Load hyperparameters
    try:
        with open(rd / "hparams.yaml") as f:
            data = yaml.load(f, Loader=yaml.CLoader)
    except FileNotFoundError:
        print(f"No hparams.yaml found in {rd}. Skipping this directory")
        continue
    data["folder"] = rd.name

    # Load performance data
    results = SummaryReader(rd).scalars
    valid_acc = results[results["tag"] == "valid_acc"]["value"]
    data["max_valid_acc"] = valid_acc.max()
    train_loss = results[results["tag"] == "train_loss"]["value"]
    data["min_train_loss"] = train_loss.min()
    valid_loss = results[results["tag"] == "valid_loss"]["value"]
    data["min_valid_loss"] = valid_loss.min()
    collected.append(data)

df = pd.DataFrame(collected)
interesting_columns = ["scale_grad", "method", "num_conv_layers", 'max_valid_acc', 'min_valid_loss', 'min_train_loss', 'folder']
interesting = df[interesting_columns]
exodus_data = interesting[interesting["method"] == "exodus"]
slayer_data = interesting[interesting["method"] == "slayer"]
