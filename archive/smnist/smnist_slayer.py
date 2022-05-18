import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from slayer_layer import SlayerLayer


class SlayerNetwork(pl.LightningModule):
    def __init__(
        self,
        tau_mem,
        spike_threshold,
        n_hidden_layers,
        learning_rate,
        width_grad,
        scale_grad,
        n_time_bins,
        encoding_dim,
        hidden_dim,
        decoding_func,
        weight_decay=0,
        init_weights_path=None,
        **kw_args,
    ):
        super().__init__()
        self.save_hyperparameters()

        neuron_params = {
            "type": "LIF",
            "theta": spike_threshold,
            "tauSr": tau_mem,
            "tauRef": tau_mem,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": n_time_bins}

        self.slayer = SlayerLayer(neuron_params, sim_params)

        self.linear_input = self.slayer.dense(encoding_dim, hidden_dim, weightScale=1)
        self.linear_hidden = nn.ModuleList(
            [self.slayer.dense(hidden_dim, hidden_dim, weightScale=1) for i in range(n_hidden_layers)]
        )
        self.linear_output = self.slayer.dense(hidden_dim, 10, weightScale=1)

    def forward(self, x):
        x = x.unsqueeze(3).unsqueeze(4).movedim(1, -1)
        x = self.slayer.spike(self.slayer.psp(self.linear_input(x)))
        for layer in self.linear_hidden:
            x = self.slayer.spike(self.slayer.psp(layer(x)))
        x = self.linear_output(x) 
        return x.squeeze().movedim(-1, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x is Batch, Time, Channels
        y_hat = self(x)
        if self.hparams.decoding_func == "sum_loss":
            y_sum = torch.sum(F.softmax(y_hat, dim=2), axis=1)
            loss = F.cross_entropy(y_sum, y)
        elif self.hparams.decoding_func == "last_ts":
            loss = F.cross_entropy(y_hat[:, -1], y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
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
            weight_decay=self.hparams.weight_decay,
        )
