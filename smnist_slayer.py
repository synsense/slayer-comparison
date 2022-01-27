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

        if init_weights_path is not None:
            loaded_state_dict = torch.load(init_weights_path)
            state_dict = {}
            for oldname, param in loaded_state_dict.items():
                if oldname.startswith("network"):
                    linear_idx = int(oldname.split(".")[1]) // 2 + 1
                    state_dict[f"linear{linear_idx}"] = param
                else:
                    state_dict[oldname] = param
            self.load(state_dict)

    def forward(self, x):
        lin1 = self.linear1(x)
        spike1 = self.slayer.spike(self.slayer.psp(lin1.movedim(1, -1))).movedim(-1, 1)
        lin2 = self.linear2(spike1)
        spike2 = self.slayer.spike(self.slayer.psp(lin2.movedim(1, -1))).movedim(-1, 1)
        out3 = self.linear2(spike2)

        return out3

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
