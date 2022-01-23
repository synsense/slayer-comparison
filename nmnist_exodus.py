import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa


class SinabsNetwork(pl.LightningModule):
    def __init__(self, 
                batch_size=None, 
                tau_mem=10., 
                spike_threshold=0.1, 
                learning_rate=1e-3, 
                weight_decay=0,
                method='exodus'
        ):
        super().__init__()
        self.save_hyperparameters()

        act_fn = sa.ActivationFunction(spike_threshold=spike_threshold,
                                        spike_fn=sa.SingleSpike,
                                        reset_fn=sa.MembraneSubtract(),
                                        surrogate_grad_fn=sa.SingleExponential(),)

        if method=='exodus':
            from sinabs.slayer.layers import LIFSqueeze, ExpLeakSqueeze
        else:
            from sinabs.layers import LIFSqueeze, ExpLeakSqueeze

        self.network = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1), # compresses Batch and Time dimension
            nn.Conv2d(2, 12, 5),
            LIFSqueeze(tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 64, 5),
            LIFSqueeze(tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 10),
            ExpLeakSqueeze(tau_leak=10*tau_mem, batch_size=batch_size),
            nn.Unflatten(0, (batch_size, -1))
        )

        if method != 'exodus':
            for layer in self.spiking_layers:
                layer.tau_mem.requires_grad = False

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.zero_grad()
        x, y = batch # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.zero_grad()
        x, y = batch # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    @property
    def spiking_layers(self):
        return [layer for layer in self.network.children() if isinstance(layer, sl.StatefulLayer)]

    @property
    def linear_layers(self):
        return [layer for layer in self.network.children() if not isinstance(layer, sl.StatefulLayer)]

    def zero_grad(self):
        for layer in self.spiking_layers:
            layer.zero_grad()

    def reset_states(self):
        for layer in self.spiking_layers:
            layer.reset_states()
