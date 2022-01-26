import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from slayer_layer import SlayerLayer


class SlayerNetwork(pl.LightningModule):
    def __init__(
        self,
        tau_mem=10.0,
        spike_threshold=0.1,
        learning_rate=1e-3,
        weight_decay=0,
        width_grad=1.0,
        scale_grad=1.0,
        n_time_bins=100,
        architecture="paper",
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

        self.architecture = architecture

        if architecture == "paper":
            self.conv1 = torch.nn.utils.weight_norm(
                self.slayer.conv(2, 12, 5, weightScale=1), name="weight"
            )
            self.pool1 = self.slayer.pool(2)
            self.conv2 = torch.nn.utils.weight_norm(
                self.slayer.conv(12, 64, 5, weightScale=1), name="weight"
            )
            self.pool2 = self.slayer.pool(2)
            self.fc1 = torch.nn.utils.weight_norm(
                self.slayer.dense((5, 5, 64), 10), name="weight"
            )

        elif architecture == "larger":
            self.conv1 = torch.nn.utils.weight_norm(
                self.slayer.conv(2, 16, 3, weightScale=1, padding=1), name="weight"
            )

            self.pool1 = self.slayer.pool(2)
            self.conv2 = torch.nn.utils.weight_norm(
                self.slayer.conv(16, 32, 3, weightScale=1, padding=1), name="weight"
            )
            self.pool2 = self.slayer.pool(2)
            self.conv3 = torch.nn.utils.weight_norm(
                self.slayer.conv(32, 64, 3, weightScale=1, padding=1), name="weight"
            )
            self.fc1 = torch.nn.utils.weight_norm(
                self.slayer.dense((8, 8, 64), 512, weightScale=1), name="weight"
            )
            self.fc2 = torch.nn.utils.weight_norm(
                self.slayer.dense(512, 10, weightScale=1), name="weight"
            )

        # Undo slayer's scaling of pooling layer
        self.pool1.weight.data /= self.pool1.weight.numel() * 1.1
        self.pool2.weight.data /= self.pool2.weight.numel() * 1.1

    def forward(self, x):
        x = x.movedim(1, -1)
        if self.architecture == "paper":
            out1 = self.pool1(self.slayer.spike(self.slayer.psp(self.conv1(x))))
            out2 = self.pool2(self.slayer.spike(self.slayer.psp(self.conv2(out1))))
            out = self.fc1(out2)
        else:
            out1 = self.pool1(self.slayer.spike(self.slayer.psp(self.conv1(x))))
            out2 = self.pool2(self.slayer.spike(self.slayer.psp(self.conv2(out1))))
            out3 = self.slayer.spike(self.slayer.psp(self.conv3(out2)))
            out4 = self.slayer.spike(self.slayer.psp(self.fc1(out3)))
            out = self.fc2(out4)

        return out.movedim(-1, 1).flatten(-3)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.sum(1), y)
        self.log("valid_loss", loss, prog_bar=True)
        prediction = y_hat.sum(1).argmax(1)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("valid_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
