import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa


class SinabsNetwork(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, tau_mem=10., spike_threshold=0.1, weight_decay=None):
        super(SinabsNetwork, self).__init__()
        self.save_hyperparameters()

        self.network = nn.Sequential(
            nn.Linear(200, 256),
            sl.LIF(tau_mem=tau_mem, activation_fn=sa.ActivationFunction(spike_threshold=spike_threshold)),
            nn.Linear(256, 200),
            sl.LIF(tau_mem=tau_mem, activation_fn=sa.ActivationFunction(spike_threshold=spike_threshold)),
        )
        
    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.zero_grad()
        x, y = batch
        x = x.movedim(2,1)
        y = y.movedim(2,1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def zero_grad(self):
        self.network[1].zero_grad()
        self.network[3].zero_grad()