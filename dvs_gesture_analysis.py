from pathlib import Path
from tbparse import SummaryReader
import yaml
import pandas as pd

# File handle
log_path = Path("lightning_logs/dvs/both")
list(log_path.iterdir())
run_dirs = log_path.iterdir()

# Iterate over runs
collected = []
for rd in run_dirs:
    # Load hyperparameters
    with open(rd / "hparams.yaml") as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    # Load performance data
    results = SummaryReader(rd).scalars
    valid_acc = results[results["tag"] == "valid_acc"]["value"]
    data["max_valid_acc"] = valid_acc.max()
    valid_loss = results[results["tag"] == "valid_loss"]["value"]
    data["min_valid_loss"] = valid_loss.min()
    collected.append(data)

df = pd.DataFrame(collected)
interesting_columns = ["scale_grad", "method", "num_conv_layers", 'max_valid_acc', 'min_valid_loss']
interesting = df[interesting_columns]
exodus_data = interesting[interesting["method"] == "exodus"]
slayer_data = interesting[interesting["method"] == "slayer"]
