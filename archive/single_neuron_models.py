import torch
import torch.nn as nn
from torch.nn import functional as F
import sinabs.activation as sa
import sinabs.exodus.layers as sel
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from slayer_layer import SlayerLayer


class SlayerNeuron(torch.nn.Module):
    def __init__(self, weight, tau_mem, spike_threshold, n_time_steps, width_grad, scale_grad):
        super().__init__()

        neuron_params = {
            "type": "CUBALIF",
            "theta": spike_threshold,
            "tauSr": tau_mem,
            "tauRef": tau_mem,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": n_time_steps}
        self.slayer = SlayerLayer(neuron_params, sim_params)
        self.lin = self.slayer.dense(1, 1)
        self.lin.weight.data *= 0
        self.lin.weight.data += weight

    def forward(self, x):
        weighted = self.lin(x.movedim(1, -1))
        psp = self.slayer.psp(weighted)
        self.psp_pre = psp.clone().movedim(-1, 1)
        out = self.slayer.spike(psp)
        self.psp_post = psp.clone().movedim(-1, 1)
        return out.movedim(-1, 1)

    @property
    def spiking_layers(self):
        return None


class ExodusNeuron(torch.nn.Module):
    def __init__(self, weight, tau_mem, spike_threshold, width_grad, scale_grad):
        super().__init__()

        act_fn = sa.ActivationFunction(
            spike_threshold=spike_threshold,
            spike_fn=sa.SingleSpike,
            reset_fn=sa.MembraneSubtract(),
            surrogate_grad_fn=sa.SingleExponential(
                grad_width=width_grad, 
                grad_scale=scale_grad
            ),
        )

        self.lin = nn.Linear(1, 1, bias=False)
        self.syn = sel.ExpLeak(tau_leak=tau_mem, norm_input=False)
        self.lif = sel.LIF(tau_mem=tau_mem, activation_fn=act_fn, norm_input=False)
        self.lin.weight.data *= 0
        self.lin.weight.data += weight

    def forward(self, x):
        self.reset_states()
        x = self.lin(x)
        x = self.syn(x)
        x = self.lif(x)
        return x

    @property
    def spiking_layers(self):
        return [self.syn, self.lif]

    def reset_states(self):
        for layer in self.spiking_layers:
            layer.reset_states()

    def zero_grad(self, set_to_none: bool = False) -> None:
        for layer in self.spiking_layers:
            layer.zero_grad()
        return super().zero_grad(set_to_none)
