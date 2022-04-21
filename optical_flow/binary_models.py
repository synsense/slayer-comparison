####
# Model definitions for binary optic flow toy task
####

import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/..")

import torch

from sinabs.exodus.layers import IAFSqueeze, LIFSqueeze
from sinabs.layers import IAFSqueeze as IAFSqueezeSinabs
from sinabs.layers import LIFSqueeze as LIFSqueezeSinabs
import sinabs.activation as sina

from slayer_layer import SlayerLayer


class ExodusModel(torch.nn.Module):
    def __init__(
        self,
        grad_width,
        grad_scale,
        num_ts,
        thr,
        neuron_type="IAF",
        tau_leak=20,
    ):
        super().__init__()

        surrogate_grad_fn = sina.SingleExponential(
            grad_width=grad_width, grad_scale=grad_scale
        )
        
        self.neuron_type = neuron_type
        
        kwargs_spiking = {
            "spike_threshold": thr,
            "min_v_mem": None,
            "reset_fn": sina.MembraneSubtract(),
            "num_timesteps": num_ts,
            "spike_fn": sina.SingleSpike,
            "surrogate_grad_fn": surrogate_grad_fn,
        }

        if self.neuron_type == "LIF":           
            kwargs_spiking["tau_mem"] = tau_leak
            kwargs_spiking["norm_input"] = False

        self.pool0 = torch.nn.AvgPool2d(4)
        self.conv0 = torch.nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=7, padding=3, bias=False
        )
        self.spk0 = self.spiking_layer_class(**kwargs_spiking)

        self.pool1 = torch.nn.AvgPool2d(4)
        self.conv1 = torch.nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=7, padding=3, bias=False
        )
        self.spk1 = spiking_layer_class(**kwargs_spiking)

        self.pool2 = torch.nn.AvgPool2d(4)
        self.linear = torch.nn.Linear(4*4*8, 2, bias=False)

    def forward(self, inp):
        
        n, c, y, x, t = inp.shape
        data = inp.movedim(-1,1).flatten(start_dim=0, end_dim=1)  # (NxT, 2, 256, 256)

        out0 = self.spk0(self.conv0(self.pool0(data)))  # (NxT, 4, 64, 64)
        out1 = self.spk1(self.conv1(self.pool1(out0)))  # (NxT, 8, 16, 16)
        out2 = self.pool2(out1).flatten(start_dim=1)  # (NxT, 8 * 4 * 4)
        out3 = self.linear(out2)  # (NxT, 2)

        self.reset()

        return out3.reshape(n, t, 2).movedim(1, -1)  # (N, 2, T)

    def reset(self):
        for lyr in (self.spk0, self.spk1):
            lyr.reset_states()
            lyr.zero_grad()

    def import_paramers(self, parameters):
        for k, p in parameters.items():
            getattr(self, k).weight.data = p.clone()

    @property
    def spiking_layer_class(self):
        if self.neuron_type == "IAF":
            return IAFSqueeze
        elif self.neuron_type == "LIF":
            return LIFSqueeze

    @property
    def parameter_copy(self):
        return {
            "conv0": self.conv0.weight.data.clone(),
            "conv1": self.conv1.weight.data.clone(),
            "linear": self.linear.weight.data.clone()
        }
    

class SinabsModel(torch.nn.Module):

    @property
    def spiking_layer_class(self):
        if self.neuron_type == "IAF":
            return IAFSqueezeSinabs
        elif self.neuron_type == "LIF":
            return LIFSqueezeSinabs


class SlayerModel(torch.nn.Module):
    def __init__(self, grad_width, grad_scale, num_ts, thr, neuron_type="IAF", tau_leak=20):
        super().__init__()

        neuron_params = {
            "type": neuron_type,
            "theta": thr,
            "scaleRef": 1,
            "tauRho": grad_width,
            "scaleRho": grad_scale,
        }
        if neuron_type == "LIF":
            neuron_params["tauSr"] = tau_leak
            neuron_params["tauRef"] = tau_leak
        sim_params = {"Ts": 1.0, "tSample": num_ts}

        self.slayer = SlayerLayer(neuron_params, sim_params)

        self.pool0 = self.slayer.pool(4)
        self.conv0 = self.slayer.conv(
            inChannels=2, outChannels=4, kernelSize=7, padding=3, stride=1
        )

        self.pool1 = self.slayer.pool(4)
        self.conv1 = self.slayer.conv(
            inChannels=4, outChannels=8, kernelSize=7, padding=3, stride=1
        )

        self.pool2 = self.slayer.pool(4)
        self.linear = self.slayer.dense((4,4,8), 2)

        # Undo slayer's scaling of pooling layer
        self.pool0.weight.data.fill_(1. / self.pool1.weight.numel())
        self.pool1.weight.data.fill_(1. / self.pool1.weight.numel())
        self.pool2.weight.data.fill_(1. / self.pool2.weight.numel())

    def forward(self, data):

        out0 = self.slayer.spike(self.slayer.psp(self.conv0(self.pool0(data))))
        out1 = self.slayer.spike(self.slayer.psp(self.conv1(self.pool1(out0))))
        out2 = self.linear(self.pool2(out1))

        return out2.squeeze(-2).squeeze(-2)

    def reset(self):
        """Dummy function to have same API as ExodusModel"""
        pass

    def import_paramers(self, parameters):
        for k, p in parameters.items():
            if k.startswith("conv"):
                getattr(self, k).weight.data = p.clone().unsqueeze(-1)
            elif k == "linear":
                self.linear.weight.data = p.clone.reshape(*self.linear.weights.shape)

    @property
    def parameter_copy(self):
        out_channels = self.linear.out_channels
        return {
            "conv0": self.conv0.weight.data.squeeze(-1).clone(),
            "conv1": self.conv1.weight.data.squeeze(-1).clone(),
            "linear": self.linear.weight.data.reshape(out_channels, -1).clone()
        }
