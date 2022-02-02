import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sa
from typing import Dict, Any
from torch.nn.utils import weight_norm
from sinabs.exodus.layers import LIFSqueeze


class ExodusNetwork(pl.LightningModule):
    def __init__(
        self,
        batch_size=None,
        tau_mem=10.0,
        spike_threshold=0.1,
        learning_rate=1e-3,
        weight_decay=0,
        width_grad=1.0,
        scale_grad=1.0,
        architecture="paper",
        init_weights_path=None,
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

        if architecture == "paper":
            self.network = nn.Sequential(
                nn.Flatten(
                    start_dim=0, end_dim=1
                ),  # compresses Batch and Time dimension
                weight_norm(nn.Conv2d(2, 12, 5, bias=False), name="weight"),
                LIFSqueeze(
                    tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size
                ),
                nn.AvgPool2d(2, ceil_mode=True),
                weight_norm(nn.Conv2d(12, 64, 5, bias=False), name="weight"),
                LIFSqueeze(
                    tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size
                ),
                nn.AvgPool2d(2, ceil_mode=True),
                nn.Flatten(),
                weight_norm(nn.Linear(64 * 8 * 8, 10, bias=False), name="weight"),
                nn.Unflatten(0, (batch_size, -1)),
            )

            if init_weights_path is not None:
                init_weights = torch.load(init_weights_path)
                self.network[1].weight_g.data = init_weights["conv1.weight_g"].squeeze(
                    -1
                )
                self.network[1].weight_v.data = init_weights["conv1.weight_v"].squeeze(
                    -1
                )
                self.network[4].weight_g.data = init_weights["conv2.weight_g"].squeeze(
                    -1
                )
                self.network[4].weight_v.data = init_weights["conv2.weight_v"].squeeze(
                    -1
                )
                self.network[8].weight_g.data = init_weights["fc1.weight_g"].flatten(1)
                self.network[8].weight_v.data = init_weights["fc1.weight_v"].flatten(1)

        elif architecture == "larger":
            self.network = nn.Sequential(
                nn.Flatten(
                    start_dim=0, end_dim=1
                ),  # compresses Batch and Time dimension
                weight_norm(nn.Conv2d(2, 16, 5, padding=1, bias=False), name="weight"),
                LIFSqueeze(
                    tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size
                ),
                nn.AvgPool2d(2, ceil_mode=True),
                weight_norm(nn.Conv2d(16, 32, 3, padding=1, bias=False), name="weight"),
                LIFSqueeze(
                    tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size
                ),
                nn.AvgPool2d(2, ceil_mode=True),
                weight_norm(nn.Conv2d(32, 64, 3, padding=1, bias=False), name="weight"),
                LIFSqueeze(
                    tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size
                ),
                nn.Flatten(),
                weight_norm(nn.Linear(8 * 8 * 64, 512, bias=False), name="weight"),
                LIFSqueeze(
                    tau_mem=tau_mem, activation_fn=act_fn, batch_size=batch_size
                ),
                weight_norm(nn.Linear(512, 10, bias=False), name="weight"),
                nn.Unflatten(0, (batch_size, -1)),
            )

            if init_weights_path is not None:
                init_weights = torch.load(init_weights_path)
                self.network[1].weight_g.data = init_weights["conv1.weight_g"].squeeze(
                    -1
                )
                self.network[1].weight_v.data = init_weights["conv1.weight_v"].squeeze(
                    -1
                )
                self.network[4].weight_g.data = init_weights["conv2.weight_g"].squeeze(
                    -1
                )
                self.network[4].weight_v.data = init_weights["conv2.weight_v"].squeeze(
                    -1
                )
                self.network[7].weight_g.data = init_weights["conv3.weight_g"].squeeze(
                    -1
                )
                self.network[7].weight_v.data = init_weights["conv3.weight_v"].squeeze(
                    -1
                )
                self.network[10].weight_g.data = init_weights["fc1.weight_g"].flatten(1)
                self.network[10].weight_v.data = init_weights["fc1.weight_v"].flatten(1)
                self.network[12].weight_g.data = init_weights["fc2.weight_g"].flatten(1)
                self.network[12].weight_v.data = init_weights["fc2.weight_v"].flatten(1)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        self.zero_grad()
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.zero_grad()
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log("valid_loss", loss, prog_bar=True)
        prediction = y_hat.sum(1).argmax(1)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("valid_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.zero_grad()
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        prediction = y_hat.sum(1).argmax(1)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("test_acc", accuracy)

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
        for name, parameter in checkpoint["state_dict"].items():
            # uninitialise states so that there aren't any problems
            # when loading the model from a checkpoint
            if "v_mem" in name or "activations" in name:
                checkpoint["state_dict"][name] = torch.zeros(
                    (0), device=parameter.device
                )
        return super().on_save_checkpoint(checkpoint)
