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
        init_weights_path=None,
        encoding_dim=80,
        hidden_dim1=128,
        hidden_dim2=256,
        decoding_func=None,
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

        self.linear1 = torch.nn.Linear(encoding_dim, hidden_dim1, bias=False)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.linear3 = torch.nn.Linear(hidden_dim2, 10, bias=False)

        self.unflatten = torch.nn.Unflatten(-1, (1, 1, -1))

        if init_weights_path is not None:
            loaded_state_dict = torch.load(init_weights_path)
            if any(k.startswith("linear") for k in loaded_state_dict.keys()):
                # Assume parameters come from slayer
                self.load_state_dict(loaded_state_dict, strict=False)
            else:
                # Assume parameters come from exodus
                state_dict = {
                    "linear1.weight": loaded_state_dict["0.weight"],
                    "linear2.weight": loaded_state_dict["2.weight"],
                    "linear3.weight": loaded_state_dict["4.weight"],
                }
                self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # Batch, Time
        x = self.unflatten(x)
        lin1 = self.linear1(x)
        spike1 = self.slayer.spike(self.slayer.psp(lin1.movedim(1, -1))).movedim(-1, 1)
        lin2 = self.linear2(spike1)
        spike2 = self.slayer.spike(self.slayer.psp(lin2.movedim(1, -1))).movedim(-1, 1)
        out3 = self.linear3(spike2)

        return out3.flatten(2, 4)

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
