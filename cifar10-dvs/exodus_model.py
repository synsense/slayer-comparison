import pytorch_lightning as pl
import sinabs.activation as sa
import sinabs.exodus.layers as sel
import sinabs.layers as sl
import torch
import torch.nn.functional as F

from resnet import sew_resnet18


class ExodusNetwork(pl.LightningModule):
    def __init__(
        self,
        tau_mem,
        batch_size,
        spike_threshold=1.0,
        learning_rate=1e-3,
        optimizer="adam",
    ):
        super().__init__()
        self.save_hyperparameters()

        kw_args = dict(
            norm_input=False,
            spike_threshold=spike_threshold,
            spike_fn=sa.MultiSpike,
            reset_fn=sa.MembraneSubtract(),
            surrogate_grad_fn=sa.SingleExponential(),
        )

        self.network = sew_resnet18(
            cnf="ADD",
            num_classes=10,
            spiking_neuron=sel.LIFSqueeze,
            batch_size=batch_size,
            tau_mem=tau_mem,
            **kw_args,
        )

    def forward(self, x):
        self.reset_states()
        x = x.flatten(0, 1)
        x = self.network(x)
        return x.unflatten(0, (self.hparams.batch_size, -1))

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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [lr_scheduler]

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
