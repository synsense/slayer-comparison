####
# Slightly more complex networks than in simple_lif.py
####

import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/..")

import torch

from sinabs.slayer.layers import LIFSqueeze, ExpLeakSqueeze
import sinabs.activation as sina

from slayer_layer import SlayerLayer


# - Network definition
class SlayerNet(torch.nn.Module):
    def __init__(self, num_timesteps, tau, thr, width_grad, scale_grad):
        super().__init__()

        neuron_params = {
            "type": "CUBALIF",
            "theta": thr,
            "tauSr": tau,
            "tauRef": tau,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": num_timesteps}
        self.slayer = SlayerLayer(neuron_params, sim_params)
        self.conv1 = self.slayer.conv(
            inChannels=2, outChannels=4, kernelSize=5, padding=2
        )
        self.pool1 = self.slayer.pool(4)
        self.conv2 = self.slayer.conv(
            inChannels=4, outChannels=8, kernelSize=3, padding=1
        )
        self.pool2 = self.slayer.pool(2)
        self.lin = self.slayer.dense((4, 4, 8), 2)

        # Undo slayer's scaling of pooling layer
        self.pool1.weight.data /= self.pool1.weight.numel() * 1.1
        self.pool2.weight.data /= self.pool2.weight.numel() * 1.1

    def forward(self, x):
        # Move time to back
        x = x.movedim(1, -1)

        # Input: (batch, 2, 32, 32, time)
        pooled1 = self.pool1(self.conv1(x))  # 4, 8, 8
        self.out1 = self.slayer.spike(self.slayer.psp(pooled1))

        pooled2 = self.pool2(self.conv2(self.out1))  # 8, 4, 4
        self.out2 = self.slayer.spike(self.slayer.psp(pooled2))

        out = self.lin(self.out2)
        # out shape before reshaping is (batch, ch, 1, 1, time)
        return out.movedim(-1, 1).squeeze(-1).squeeze(-1)


class ExodusNet(torch.nn.Module):
    def __init__(self, num_timesteps, tau, thr, width_grad, scale_grad):
        super().__init__()

        # activation function
        activation = sina.ActivationFunction(
            spike_threshold=thr,
            spike_fn=sina.SingleSpike,
            reset_fn=sina.MembraneSubtract(),
            surrogate_grad_fn=sina.SingleExponential(
                beta=width_grad, grad_scale=scale_grad
            ),
        )

        # No biases for conv and lin layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=5, padding=2, bias=False
        )
        self.pool1 = torch.nn.AvgPool2d(4)
        self.syn1 = ExpLeakSqueeze(tau_leak=tau, num_timesteps=num_timesteps)
        self.spk1 = LIFSqueeze(
            tau_mem=tau,
            threshold_low=None,
            activation_fn=activation,
            num_timesteps=num_timesteps,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False
        )
        self.pool2 = torch.nn.AvgPool2d(2)
        self.syn2 = ExpLeakSqueeze(tau_leak=tau, num_timesteps=num_timesteps)
        self.spk2 = LIFSqueeze(
            tau_mem=tau,
            threshold_low=None,
            activation_fn=activation,
            num_timesteps=num_timesteps,
        )
        self.lin = torch.nn.Linear(4 * 4 * 8, 2, bias=False)

        self.num_timesteps = num_timesteps

    def forward(self, x):
        batch, time, *__ = x.shape
        # Input: (batch, time, 2, 32, 32)
        pooled1 = self.pool1(self.conv1(x.flatten(start_dim=0, end_dim=1)))  # 8, 8, 4
        self.out1 = self.spk1(self.syn1(pooled1))

        pooled2 = self.pool2(self.conv2(self.out1))  # 4, 4, 8
        self.out2 = self.spk2(self.syn2(pooled2))

        out = self.lin(self.out2.flatten(-3))

        # Slayer does not store states between forward calls
        self.reset()

        return out.reshape(batch, time, -1)

    def reset(self):
        self.syn1.reset_states()
        self.syn2.reset_states()
        self.spk1.reset_states()
        self.spk2.reset_states()


def transfer_weights(src_slr, tgt_exo):
    # Slayer conv weights have additional dimension for time (with size 1)
    tgt_exo.conv1.weight.data = src_slr.conv1.weight.detach().squeeze(-1)
    tgt_exo.conv2.weight.data = src_slr.conv2.weight.detach().squeeze(-1)
    # Slayer lin weights have additional dimensions for input height / width
    tgt_exo.lin.weight.data = src_slr.lin.weight.detach().flatten(start_dim=1)


num_timesteps = 100
num_batches = 2
tau = 10.0
thr = 1
width_grad = 1
scale_grad = 1

cont = True
# Try different inputs and network instances
for j in range(10):
    if cont:
        slyr = SlayerNet(num_timesteps, tau, thr, width_grad, scale_grad).cuda()
        exod = ExodusNet(num_timesteps, tau, thr, width_grad, scale_grad).cuda()
        transfer_weights(slyr, exod)

        for i in range(4):
            inp = torch.rand((num_batches, num_timesteps, 2, 32, 32)).cuda()
            out_s = slyr(inp)
            out_e = exod(inp)

            print("Out sum:", out_s.sum().item())
            if not torch.allclose(out_s, out_e, atol=5e-4, rtol=1e-5):
                print("\tMax abs diff:", torch.max(torch.abs(out_s - out_e)).item())
                print("\tMean abs diff:", torch.mean(torch.abs(out_s - out_e)).item())
                print("\tMean diff:", torch.mean(out_s - out_e).item())
                print(
                    "\tRMS diff:", torch.sqrt(torch.mean((out_s - out_e) ** 2)).item()
                )
                print(
                    "\tNum different:",
                    torch.sum(
                        (
                            torch.isclose(out_s, out_e, atol=5e-4, rtol=1e-5) == False
                        ).float()
                    ).item(),
                )
                cont = False  # Break outer loop
                break
