import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa
from typing import Dict, Any


class SinabsNetwork(pl.LightningModule):
    def __init__(
        self,
        tau_mem=10.0,
        spike_threshold=0.1,
        learning_rate=1e-3,
        weight_decay=0,
        method="exodus",
        encoding_dim=250,
        hidden_dim=25,
    ):
        super().__init__()
        self.save_hyperparameters()

        act_fn = sa.ActivationFunction(
            spike_threshold=spike_threshold,
            spike_fn=sa.SingleSpike,
            reset_fn=sa.MembraneSubtract(),
            surrogate_grad_fn=sa.SingleExponential(),
        )

        if method == "exodus":
            from sinabs.slayer.layers import LIF, ExpLeakSqueeze
        else:
            from sinabs.layers import LIF, ExpLeakSqueeze

        self.network = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim, bias=False),
            LIF(tau_mem=tau_mem, activation_fn=act_fn),
            nn.Linear(hidden_dim, 1, bias=False),
            LIF(tau_mem=tau_mem, activation_fn=act_fn),
        )

        if method != "exodus":
            for layer in self.spiking_layers:
                layer.tau_mem.requires_grad = False

    def forward(self, x):
        return self.network(x*1e8)

    def training_step(self, batch, batch_idx):
        self.zero_grad()
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        # import ipdb; ipdb.set_trace()
        loss = F.mse_loss(y_hat.squeeze(2), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @property
    def spiking_layers(self):
        return [
            layer
            for layer in self.network.children()
            if isinstance(layer, sl.StatefulLayer)
        ]

    @property
    def weight_layers(self):
        return [
            layer
            for layer in self.network.children()
            if not isinstance(layer, sl.StatefulLayer)
        ]

    def zero_grad(self):
        for layer in self.spiking_layers:
            layer.zero_grad()

    def reset_states(self):
        for layer in self.spiking_layers:
            layer.reset_states()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, parameter in checkpoint['state_dict'].items():
            # uninitialise states so that there aren't any problems 
            # when loading the model from a checkpoint
            if 'v_mem' in name or 'activations' in name:
                checkpoint['state_dict'][name] = torch.zeros((0), device=parameter.device)
        return super().on_save_checkpoint(checkpoint)
