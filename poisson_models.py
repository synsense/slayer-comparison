import torch
import torch.nn as nn
from torch.nn import functional as F
import sinabs.activation as sa
from sinabs.slayer.layers import LIF
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from slayer_layer import SlayerLayer


class SlayerNet(torch.nn.Module):
    def __init__(self, encoding_dim, hidden_dim, tau_mem, spike_threshold, n_time_steps, width_grad, scale_grad):
        super().__init__()

        neuron_params = {
            "type": "LIF",
            "theta": spike_threshold,
            "tauSr": tau_mem,
            "tauRef": tau_mem,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": n_time_steps}
        self.slayer = SlayerLayer(neuron_params, sim_params)
        self.lin1 = self.slayer.dense(encoding_dim, hidden_dim)
        self.lin2 = self.slayer.dense(hidden_dim, 1)

    def forward(self, x):
        weighted1 = self.lin1(x.movedim(1, -1))
        psp = self.slayer.psp(weighted1)
        self.psp_pre1 = psp.clone().movedim(-1, 1)
        out1 = self.slayer.spike(psp)
        self.psp_post1 = psp.clone().movedim(-1, 1)

        weighted2 = self.lin2(out1)
        psp = self.slayer.psp(weighted2)
        self.psp_pre2 = psp.clone().movedim(-1, 1)
        out2 = self.slayer.spike(psp)
        self.psp_post2 = psp.clone().movedim(-1, 1)
        
        return out2.movedim(-1, 1)

    @property
    def spiking_layers(self):
        return None



class ExodusNet(torch.nn.Module):
    def __init__(self, encoding_dim, hidden_dim, tau_mem, spike_threshold, n_time_steps, width_grad, scale_grad):
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
        neuron_params = {
            "type": "LIF",
            "theta": spike_threshold,
            "tauSr": tau_mem,
            "tauRef": tau_mem,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": n_time_steps}
        self.slayer = SlayerLayer(neuron_params, sim_params)
        self.lin1 = self.slayer.dense(encoding_dim, hidden_dim)
        self.lin2 = self.slayer.dense(hidden_dim, 1)

        self.lif1 = LIF(tau_mem=tau_mem, threshold_low=None, activation_fn=act_fn)
        self.lif2 = LIF(tau_mem=tau_mem, threshold_low=None, activation_fn=act_fn)

    def forward(self, x):
        self.reset_states()
        weighted1 = self.lin1(x.movedim(1, -1)).movedim(-1, 1)
        out1 = self.lif1(weighted1)
        weighted2 = self.lin2(out1.movedim(1, -1)).movedim(-1, 1)
        out2 = self.lif2(weighted2)
        return out2

    @property
    def spiking_layers(self):
        return [self.lif1, self.lif2]

    def reset_states(self):
        for layer in self.spiking_layers:
            layer.reset_states()


def smooth(x,window_len=11,window='hanning'):
        if window_len<3:
                return x
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]