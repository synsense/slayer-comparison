import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa
from typing import Dict, Any
import sinabs.exodus.layers as ssl
from torch.nn.utils import weight_norm


class Memory(nn.Sequential):
    def __init__(self, encoding_dim, output_dim, activation, tau_mem):
        super().__init__(
            nn.Linear(encoding_dim, output_dim, bias=False),
            ssl.LIF(tau_mem=tau_mem, activation_fn=activation),
        )


class ExodusNetwork(pl.LightningModule):
    def __init__(
        self,
        tau_mem,
        tau_syn,
        spike_threshold,
        learning_rate,
        width_grad,
        scale_grad,
        encoding_dim,
        hidden_dim,
        decoding_func,
        init_weights_path,
        n_hidden_layers,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        act_fn = sa.ActivationFunction(
            spike_threshold=spike_threshold,
            spike_fn=sa.SingleSpike,
            reset_fn=sa.MembraneSubtract(),
            surrogate_grad_fn=sa.SingleExponential(
                grad_width=width_grad, grad_scale=scale_grad
            ),
        )

        self.network = nn.Sequential(
            Memory(encoding_dim, hidden_dim, act_fn, tau_mem),
            *[
                Memory(hidden_dim, hidden_dim, act_fn, tau_mem)
                for i in range(n_hidden_layers - 1)
            ],
            nn.Linear(hidden_dim, 10, bias=False),
        )

        if init_weights_path:
            self.network.load_state_dict(torch.load(init_weights_path))

        self.activations = {}
        for layer in self.spiking_layers:
            layer.register_forward_hook(self.save_activations)

    def forward(self, x):
        return self.network(x)

    def save_activations(self, module, input, output):
        self.activations[module] = output

    def training_step(self, batch, batch_idx):
        self.reset_states()
        self.zero_grad()
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        firing_rate = torch.cat(list(self.activations.values())).mean()
        self.log("firing_rate", firing_rate, prog_bar=True)
        if self.hparams.decoding_func == "sum_loss":
            y_sum = torch.sum(F.softmax(y_hat, dim=2), axis=1)
            loss = F.cross_entropy(y_sum, y)
        elif self.hparams.decoding_func == "last_ts":
            loss = F.cross_entropy(y_hat[:, -1], y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.reset_states()
        self.zero_grad()
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        if self.hparams.decoding_func == "sum_loss":
            y_sum = torch.sum(F.softmax(y_hat, dim=2), axis=1)
            loss = F.cross_entropy(y_sum, y)
            prediction = y_sum.argmax(1)
        elif self.hparams.decoding_func == "last_ts":
            loss = F.cross_entropy(y_hat[:, -1], y)
            prediction = y_hat[:, -1].argmax(1)
        self.log("valid_loss", loss, prog_bar=True)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("valid_acc", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.reset_states()
        self.zero_grad()
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        if self.hparams.decoding_func == "sum_loss":
            y_sum = torch.sum(F.softmax(y_hat, dim=2), axis=1)
            loss = F.cross_entropy(y_sum, y)
            prediction = y_sum.argmax(1)
        elif self.hparams.decoding_func == "last_ts":
            loss = F.cross_entropy(y_hat[:, -1], y)
            prediction = y_hat[:, -1].argmax(1)
        self.log("test_loss", loss, prog_bar=True)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("test_acc", accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

    @property
    def sinabs_layers(self):
        return [
            layer
            for layer in self.network.modules()
            if isinstance(layer, sl.StatefulLayer)
        ]

    @property
    def spiking_layers(self):
        return [layer for layer in self.sinabs_layers if hasattr(layer, "threshold")]

    @property
    def weight_layers(self):
        return [
            layer
            for layer in self.network.modules()
            if not isinstance(layer, sl.StatefulLayer)
        ]

    def zero_grad(self):
        for layer in self.sinabs_layers:
            layer.zero_grad()

    def reset_states(self):
        for layer in self.sinabs_layers:
            layer.reset_states()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, parameter in checkpoint["state_dict"].items():
            # uninitialise states so that there aren't any problems
            # when loading the model from a checkpoint
            if "v_mem" in name or "activations" in name:
                checkpoint["state_dict"][name] = torch.zeros(
                    (0), device=parameter.device
                )
        return super().on_save_checkpoint(checkpoint)