import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from data.oxford_dataset import OxfordDataset
from sinabs_network import SinabsNetwork

voltage_decay = 0.1
tau_mem = -1.0 / np.log(1 - voltage_decay)
threshold = 0.1
learning_rate = 1e-3
weight_decay = 1e-5
print(f"tau_mem: {tau_mem}")

model = SinabsNetwork(
    tau_mem=tau_mem,
    spike_threshold=threshold,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
)

training_set = OxfordDataset()
train_loader = DataLoader(training_set, batch_size=1)

trainer = pl.Trainer(gpus=1)  # .from_argparse_args(args)
trainer.fit(model, train_loader)
