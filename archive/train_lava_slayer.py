import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.oxford_dataset import OxfordDataset
from lava_network import LavaNetwork

voltage_decay = 0.1
threshold = 0.1
learning_rate = 1e-3
weight_decay = 1e-5

model = LavaNetwork(voltage_decay=voltage_decay, spike_threshold=threshold, learning_rate=learning_rate, weight_decay=weight_decay)

training_set = OxfordDataset()
train_loader = DataLoader(training_set, batch_size=1)

trainer = pl.Trainer(gpus=1)#.from_argparse_args(args)
trainer.fit(model, train_loader)
