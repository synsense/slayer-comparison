####
# Model definitions for binary optic flow toy task
####

import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/..")

import torch

from sinabs.exodus.layers import ExpLeakSqueeze, IAFSqueeze
import sinabs.activation as sina

from slayer_layer import SlayerLayer


class ExodusModel(torch.nn.Module):
	def __init__(self, grad_width, grad_scale, num_ts, thr):
		super().__init__()

		activation_fn = sina.ActivationFunction(
			spike_threshold=thr,
			spike_fn=sina.SingleSpike,
			reset_fn=sina.MembraneSubtract(),
			surrogate_grad_fn=sina.SingleExponential(
				beta=grad_width, grad_scale=grad_scale
			)
		)

		self.pool0 = torch.nn.AvgPool2d(4)
		self.conv0 = torch.nn.Conv2d(
			in_channels=2, out_channels=4, kernel_size=7, padding=3, bias=False
		)
		self.spk0 = IAFSqueeze(
			threshold_low=None, activation_fn=activation, num_timesteps=num_ts
		)

		self.pool1 = torch.nn.AvgPool2d(4)
		self.conv1 = torch.nn.Conv2d(
			in_channels=4, out_channels=8, kernel_size=7, padding=3, bias=False
		)
		self.spk1 = IAFSqueeze(
			threshold_low=None, activation_fn=activation, num_timesteps=num_ts
		)

		self.pool2 = torch.nn.AvgPool2d(4)
		self.linear = torch.nn.linear(4*4*8, 2, bias=False)

	def forward(self, inp):
		
		data = inp.flatten(start_dim=0, end_dim=1)  # (NxT, 2, 256, 256)

        out0 = self.spk0(self.conv0(self.pool0(data)))  # (NxT, 4, 64, 64)
		out1 = self.spk1(self.conv1(self.pool1(out0)))  # (NxT, 8, 16, 16)
		out2 = self.pool2(out1).flatten(start_dim=1)  # (NxT, 8 * 4 * 4)
		out3 = self.linear(out2)  # (NxT, 2)

		self.reset()

		return out3.reshape(*inp.shape[:2], 2)  # (N, T, 2)

	def reset(self):
		for lyr in (self.spk0, self.spk1, self.spk2):
			lyr.reset_states()
			lyr.zero_grad()

class SlayerModel(torch.nn.Module):
	def __init__(self, grad_width, grad_scale, num_ts, thr):
		super().__init__()

        neuron_params = {
            "type": "IAF",
            "theta": thr,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": num_ts}

        self.slayer = SlayerLayer(neuron_params, sim_params)

		self.pool0 = self.slayer.pool(4)
		self.conv0 = self.slayer.conv(
			inChannels=2, outChannels=4, kernelSize=7, padding=3
		)
		
		self.pool1 = self.slayer.pool(4)
		self.conv1 = self.slayer.conv(
			inChannels=4, outChannels=8, kernelSize=7, padding=3
		)
		
		self.pool2 = self.slayer.pool(4)
		self.linear = self.slayer.dense((4,4,4), 2)

        # Undo slayer's scaling of pooling layer
        self.pool0.weight.data = 1. / self.pool1.weight.numel()
        self.pool1.weight.data = 1. / self.pool1.weight.numel()
        self.pool2.weight.data = 1. / self.pool2.weight.numel()

	def forward(self, inp):

		data = inp.movedim(1, -1)

		out0 = self.slayer.spike(self.slayer.psp(self.conv0(self.pool0(data))))
		out1 = self.slayer.spike(self.slayer.psp(self.conv1(self.pool1(out0))))
		out2 = self.linear(self.pool2(out1))

		return out2.movedim(-1, 1).squeeze(-1).squeeze(-1)

