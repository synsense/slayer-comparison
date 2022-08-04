import pytorch_lightning as pl
import sinabs.activation as sa
import sinabs.exodus.layers as sel
import sinabs.layers as sl
import torch
import torch.nn.functional as F

from resnet import sew_resnet18
from resnet2 import SEWResNet

class ExodusNetwork(pl.LightningModule):
    def __init__(
        self,
        tau_mem,
        batch_size,
        spike_threshold=1.0,
        learning_rate=1e-3,
        optimizer="adam",
        lr_scheduler_t_max=0,
    ):
        super().__init__()
        self.save_hyperparameters()

        spike_args = dict(
            tau_mem=tau_mem,
            norm_input=False,
            spike_threshold=spike_threshold,
            spike_fn=sa.MultiSpike,
            reset_fn=sa.MembraneSubtract(),
            surrogate_grad_fn=sa.SingleExponential(),
        )

        self.network = SEWResNet("ADD", **spike_args)

    def forward(self, x):
        self.reset_states()
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        y_decoded = y_hat.sum(1)
        loss = F.cross_entropy(y_decoded, y)
        self.log("loss/training", loss)
        for l, layer in enumerate(self.sinabs_layers):
            self.log(f"firing_rate/layer-{l}", layer.firing_rate)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        y_decoded = y_hat.sum(1)
        loss = F.cross_entropy(y_decoded, y)
        prediction = y_decoded.argmax(1)
        self.log("loss/validation", loss, prog_bar=True)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("accuracy/validation", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch  # x is Batch, Time, Channels, Height, Width
        y_hat = self(x)
        y_decoded = y_hat.sum(1)
        prediction = y_decoded.argmax(1)
        accuracy = (prediction == y).float().sum() / len(prediction)
        self.log("accuracy/test", accuracy)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
            )
        if self.hparams.lr_scheduler_t_max > 0:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.lr_scheduler_t_max)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    @property
    def sinabs_layers(self):
        return [
            layer
            for layer in self.network.modules()
            if isinstance(layer, sl.StatefulLayer)
        ]

    def reset_states(self):
        for layer in self.sinabs_layers:
            layer.reset_states()
