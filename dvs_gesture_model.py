import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import sinabs.layers as sl
import sinabs.activation as sina
from typing import Dict, Any
from torch.nn.utils import weight_norm
from sinabs.exodus.layers import LIFSqueeze, IAFSqueeze
from slayer_layer import SlayerLayer

class SlayerNetwork(nn.Module):
    def __init__(
        self,
        batch_size=None,
        tau_mem=10.0,
        spike_threshold=0.1,
        base_channels=8,
        kernel_size=3,
        num_conv_layers=4,
        width_grad=1.0,
        scale_grad=1.0,
        iaf=False,
        num_timesteps=300,
        dropout=False,
        batchnorm=False,
    ):

        super().__init__()

        neuron_params = {
            "type": "IAF" if iaf else "LIF",
            "theta": spike_threshold,
            "tauSr": tau_mem,
            "tauRef": tau_mem,
            "scaleRef": 1.,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": num_timesteps}

        self.slayer = SlayerLayer(neuron_params, sim_params)
        
        # Convolutional and linear layers
        padding = (kernel_size - 1) // 2
        in_channels = 2
        
        self.conv_layers = nn.ModuleList()
        for i in range(4):
            out_channels = base_channels * 2**i
            self.conv_layers.append(
                weight_norm(
                    self.slayer.conv(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                    ),
                    name="weight"
                )
            )
            in_channels = out_channels

        if num_conv_layers < 4:
            raise(ValueError(f"Need at least 4 conv layers"))

        for i in range(num_conv_layers - 4):
            self.conv_layers.append(
                weight_norm(
                    self.slayer.conv(in_channels, in_channels, kernel_size, padding=padding),
                    name="weight",
                )
            )
        
        self.lin = weight_norm(
            self.slayer.dense((4, 4, in_channels), 11, weightScale=1),
            name="weight"
        )
        
        # Pooling
        self.pool = self.slayer.pool(2)
        self.pool.weight.data.fill_(1.0 / self.pool.weight.numel())

        # Dropout
        self.dropout05 = nn.Dropout(0.5) if dropout else nn.Identity()
        self.dropout01 = nn.Dropout(0.1) if dropout else nn.Identity()

        # Batchnorm
        self.batchnorms = nn.ModuleList(
            nn.BatchNorm3d(conv.out_channels) if batchnorm else nn.Identity()
            for conv in self.conv_layers
        )

    def forward(self, x):
        x = x.movedim(1, -1)
        for conv, bn in zip(self.conv_layers, self.batchnorms):
            x = self.slayer.spike(self.slayer.psp(bn(conv(x))))
            x = self.dropout01(x)
            if x.shape[-2] > 4:
                x = self.pool(x)
        x = self.dropout05(x)
        out = self.lin(x)
        return out.movedim(-1, 1).flatten(-3)

    def import_parameters(self, parameters):
        for new_p, lyr in zip(parameters["conv_g"], self.conv_layers):
            lyr.weight_g.data = new_p.unsqueeze(-1).clone()
        for new_p, lyr in zip(parameters["conv_v"], self.conv_layers):
            lyr.weight_v.data = new_p.unsqueeze(-1).clone()

        self.lin.weight_g.data = parameters["lin_g"][0].reshape(*self.lin.weight_g.shape).clone()
        self.lin.weight_v.data = parameters["lin_v"][0].reshape(*self.lin.weight_v.shape).clone()


    @property
    def parameter_copy(self):
        out_channels = self.lin.out_channels
        return {
            "conv_g": [lyr.weight_g.data.squeeze(-1).clone() for lyr in self.conv_layers],
            "conv_v": [lyr.weight_v.data.squeeze(-1).clone() for lyr in self.conv_layers],
            "lin_g": [self.lin.weight_g.data.reshape(out_channels, -1).clone()],
            "lin_v": [self.lin.weight_v.data.reshape(out_channels, -1).clone()],
        }


class ExodusNetwork(nn.Module):
    def __init__(
        self,
        batch_size=None,
        tau_mem=10.0,
        spike_threshold=0.1,
        base_channels=8,
        kernel_size=3,
        num_conv_layers=4,
        width_grad=1.0,
        scale_grad=1.0,
        iaf=False,
        num_timesteps=None, # Not needed in this class. Only for compatible API
        dropout=False,
        batchnorm=False,
    ):

        super().__init__()

        spike_fn=sina.SingleSpike
        surrogate_grad_fn=sina.SingleExponential(
                grad_width=width_grad, grad_scale=scale_grad
            )
        spk_kwargs = dict(
            spike_threshold=spike_threshold,
            spike_fn=spike_fn, 
            surrogate_grad_fn=surrogate_grad_fn, 
            batch_size=batch_size,
        )
        if not iaf:
            spk_kwargs["norm_input"] = False
            spk_kwargs["tau_mem"] = tau_mem

        Spk = IAFSqueeze if iaf else LIFSqueeze
        
        # Convolutional and linear layers
        padding = (kernel_size - 1) // 2
        in_channels = 2
        
        self.conv_layers = nn.ModuleList()
        for i in range(4):
            out_channels = base_channels * 2**i
            self.conv_layers.append(
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        bias=False
                    ),
                    name="weight"
                )
            )
            in_channels = out_channels

        if num_conv_layers < 4:
            raise(ValueError(f"Need at least 4 conv layers"))

        for i in range(num_conv_layers - 4):
            self.conv_layers.append(
                weight_norm(
                    nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=False),
                    name="weight",
                )
            )
        
        self.lin = weight_norm(nn.Linear(in_channels*4*4, 11, bias=False), name="weight")
        
        # Pooling
        self.pool = nn.AvgPool2d(2)
        
        # Spiking layers
        self.spk_layers = nn.ModuleList(Spk(**spk_kwargs) for i in range(num_conv_layers))

        # Dropout
        self.dropout05 = nn.Dropout(0.5) if dropout else nn.Identity()
        self.dropout01 = nn.Dropout(0.1) if dropout else nn.Identity()

        # Batchnorm
        self.batchnorms = nn.ModuleList(
            nn.BatchNorm2d(conv.out_channels) if batchnorm else nn.Identity()
            for conv in self.conv_layers
        )

    def forward(self, x):
        batch_size, *__ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)
        for conv, spk, bn in zip(self.conv_layers, self.spk_layers, self.batchnorms):
            x = spk(bn(conv(x)))
            x = self.dropout01(x)
            if x.shape[-1] > 4:
                x = self.pool(x)

        x = self.dropout05(x)
        out = self.lin(x.flatten(start_dim=1))
        return out.reshape(batch_size, -1, *out.shape[1:])

    def import_parameters(self, parameters):
        for new_p, lyr in zip(parameters["conv_g"], self.conv_layers):
            lyr.weight_g.data = new_p.clone()
        for new_p, lyr in zip(parameters["conv_v"], self.conv_layers):
            lyr.weight_v.data = new_p.clone()

        self.lin.weight_g.data = parameters["lin_g"][0].clone()
        self.lin.weight_v.data = parameters["lin_v"][0].clone()

    @property
    def parameter_copy(self):
        return {
            "conv_g": [lyr.weight_g.data.clone() for lyr in self.conv_layers],
            "conv_v": [lyr.weight_v.data.clone() for lyr in self.conv_layers],
            "lin_g": [self.lin.weight_g.data.clone()],
            "lin_v": [self.lin.weight_v.data.clone()],
        }


class GestureNetwork(pl.LightningModule):
    def __init__(
        self,
        method,
        batch_size=None,
        tau_mem=10.0,
        spike_threshold=0.1,
        learning_rate=1e-3,
        base_channels=8,
        kernel_size=3,
        num_conv_layers=4,
        weight_decay=0,
        width_grad=1.0,
        scale_grad=1.0,
        init_weights_path=None,
        iaf=False,
        num_timesteps=300,
        optimizer="Adam",
        dropout=False,
        batchnorm=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if method == "exodus":
            net_class = ExodusNetwork
        elif method == "slayer":
            net_class = SlayerNetwork
        else:
            raise ValueError(f"Method '{method}' not supported")

        self.network = net_class(
            batch_size=batch_size,
            tau_mem=tau_mem,
            spike_threshold=spike_threshold,
            width_grad=width_grad,
            scale_grad=scale_grad,
            iaf=iaf,
            kernel_size=kernel_size,
            base_channels=base_channels,
            num_conv_layers=num_conv_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        self.optimizer_class = optimizer


    def forward(self, x):
        return self.network(x)


    def training_step(self, batch, batch_idx):
        self.reset_states()
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log("train_loss", loss)
        return loss

    def on_after_backward(self):
        grads = self.named_trainable_parameter_grads
        grad_metrics = [
            {"grads_max_" + n: torch.max(torch.abs(g)) for n, g in grads.items()},
            {"grads_max_" + n: torch.max(torch.abs(g)) for n, g in grads.items()},
            {"grads_std_" + n: torch.std(torch.abs(g)) for n, g in grads.items()},
            {"grads_mean_" + n: torch.mean(g) for n, g in grads.items()},
            {"grads_mean_abs_" + n: torch.mean(torch.abs(g)) for n, g in grads.items()},
            # "grads_norm": {n: torch.linalg.norm(g) for n, g in grads.items()},
            # "grads_std": {n: torch.std(torch.abs(g)) for n, g in grads.items()},
            # "grads_mean": {n: torch.mean(g) for n, g in grads.items()},
            # "grads_mean_abs": {n: torch.mean(torch.abs(g)) for n, g in grads.items()},
            # "grads_norm": {n: torch.linalg.norm(g) for n, g in grads.items()},
        ]
        for metric in grad_metrics:
            for k, v in metric.items():
                self.log(k, v)


    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        for name, params in self.named_trainable_parameters.items():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
            # self.logger.experiment.add_histogram(
            #     "grads_" + name, params.grad, self.current_epoch
            # )

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
        optimizer = getattr(torch.optim, self.optimizer_class)
        return optimizer(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @property
    def spiking_layers(self):
        return [
            layer
            for layer in self.network.modules()
            if isinstance(layer, sl.StatefulLayer)
        ]

    @property
    def weight_layers(self):
        return [
            layer
            for layer in self.network.modules()
            if not isinstance(layer, sl.StatefulLayer)
        ]

    @property
    def named_trainable_parameter_grads(self):
        return {
            k: p.grad for k, p in self.network.named_parameters() if p.requires_grad
        }

    @property
    def named_trainable_parameters(self):
        return {
            k: p for k, p in self.network.named_parameters() if p.requires_grad
        }

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
