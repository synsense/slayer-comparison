import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa
import lava.lib.dl.slayer as slayer


class SinabsNetwork(pl.LightningModule):
    def __init__(self, 
                tau_mem=10., 
                spike_threshold=0.1, 
                learning_rate=1e-5, 
                weight_decay=None,
                method='exodus'
        ):
        super(SinabsNetwork, self).__init__()
        self.save_hyperparameters()

        act_fn = sa.ActivationFunction(spike_threshold=spike_threshold,
                                        spike_fn=sa.SingleSpike,
                                        reset_fn=sa.MembraneReset())

        if method=='exodus':
            from sinabs.exodus.layers import LIF
        else:
            from sinabs.layers import LIF

        self.network = nn.Sequential(
            nn.Linear(200, 200, bias=False),
            LIF(tau_mem=tau_mem, activation_fn=act_fn),
            # nn.Linear(256, 200, bias=False),
            # LIF(tau_mem=tau_mem, activation_fn=act_fn),
        )

        for layer in self.spiking_layers:
            layer.tau_mem.requires_grad = False

        self.error = slayer.loss.SpikeTime(time_constant=2, filter_order=2)
        
    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.zero_grad()
        x, y = batch
        x = x.movedim(2,1)
        y = y.movedim(2,1)
        x = x/0.1
        y_hat = self(x)
        loss = self.error(y_hat, y)
        self.log('train_loss', loss)
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
