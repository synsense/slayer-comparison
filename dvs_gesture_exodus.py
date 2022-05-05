import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sina
from typing import Dict, Any
from torch.nn.utils import weight_norm
from sinabs.exodus.layers import LIFSqueeze, IAFSqueeze


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
        init_weights_path=None,
        iaf=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        spike_fn=sina.SingleSpike
        surrogate_grad_fn=sina.SingleExponential(
                grad_width=width_grad, grad_scale=scale_grad
            )
        spk_kwargs = dict(
            spike_fn=spike_fn, 
            surrogate_grad_fn=surrogate_grad_fn, 
            batch_size=batch_size,
        )
        if not iaf:
            spk_kwargs["norm_input"] = False
            spk_kwargs["tau_mem"] = False

        Spk = IAFSqueeze if iaf else LIFSqueeze
            
        self.conv1 = weight_norm(nn.Conv2d(2, 12, 5, padding=2, bias=False), name="weight")
        self.spk1 = Spk(**spk_kwargs)
        self.pool1 = nn.AvgPool2d(2, ceil_mode=True)
        self.conv2 = weight_norm(nn.Conv2d(12, 32, 5, padding=2, bias=False), name="weight")
        self.spk2 = Spk(**spk_kwargs)
        self.pool2 = nn.AvgPool2d(2, ceil_mode=True)
        self.conv3 = weight_norm(nn.Conv2d(32, 64, 5, padding=2, bias=False), name="weight")
        self.spk3 = Spk(**spk_kwargs)
        self.pool3 = nn.AvgPool2d(2, ceil_mode=True)
        self.conv4 = weight_norm(nn.Conv2d(64, 128, 5, bias=False), name="weight")
        self.spk4 = Spk(**spk_kwargs)
        self.pool4 = nn.AvgPool2d(2, ceil_mode=True)
        # self.conv5 = weight_norm(nn.Conv2d(128, 256, 5, bias=False), name="weight")
        # self.spk5 = Spk(**spk_kwargs)
        # self.conv6 = weight_norm(nn.Conv2d(256, 512, 5, bias=False), name="weight")
        # self.spk6 = Spk(**spk_kwargs)
        self.lin = weight_norm(nn.Linear(128*2*2, 11, bias=False), name="weight")

    def forward(self, x):
        batch_size, *__ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)
        out1 = self.pool1(self.spk1(self.conv1(x)))
        out2 = self.pool2(self.spk2(self.conv2(out1)))
        out3 = self.pool3(self.spk3(self.conv3(out2)))
        out4 = self.pool4(self.spk4(self.conv4(out3)))
        out = self.lin(out4.flatten(start_dim=1))
        return out.reshape(batch_size, -1, *out.shape[1:])

    def training_step(self, batch, batch_idx):
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
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
            for layer in self.children()
            if isinstance(layer, sl.StatefulLayer)
        ]

    @property
    def weight_layers(self):
        return [
            layer
            for layer in self.children()
            if not isinstance(layer, sl.StatefulLayer)
        ]

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
