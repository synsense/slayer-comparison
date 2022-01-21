import torch

from sinabs.slayer.layers import LIF
import sinabs.activation as sina

from slayer_layer import SlayerLayer


# - Network definition
class SlayerNet(torch.nn.Module):
    def __init__(self, num_timesteps, tau, thr, width_grad, scale_grad):
        super().__init__()

        neuron_params = {
            "type": "LIF",
            "theta": thr,
            "tauSr": tau,
            "tauRef": tau,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": num_timesteps}
        self.slayer = SlayerLayer(neuron_params, sim_params)
        self.lin = self.slayer.dense((4, 4, 2), 1)

    def forward(self, x):
        self.weighted = self.lin(x)
        psp = self.slayer.psp(self.weighted)
        self.psp_pre = psp.clone()
        out = self.slayer.spike(psp)
        self.psp_post = psp.clone()

        return out


class ExodusNet(torch.nn.Module):
    def __init__(self, num_timesteps, tau, thr, width_grad, scale_grad):
        super().__init__()

        neuron_params = {
            "theta": thr,
            "tauSr": tau,
            "tauRef": tau,
            "scaleRef": 1,
            "tauRho": width_grad,
            "scaleRho": scale_grad,
        }
        sim_params = {"Ts": 1.0, "tSample": num_timesteps}
        self.slayer = SlayerLayer.layer(neuron_params, sim_params)
        self.lin = self.slayer.dense((4, 4, 2), 1)

        # activation function
        activation = sina.ActivationFunction(
            spike_threshold=thr,
            spike_fn=sina.SingleSpike,
            reset_fn=sina.MembraneSubtract,
            surrogate_grad_fn=sina.SingleExponential(
                beta=width_grad, grad_scale=scale_grad
            ),
        )

        self.spk = LIF(tau_mem=tau, threshold_low=None, activation_fn=activation)

    def forward(self, x):
        self.weighted = self.lin(x)
        out = self.spk(self.weighted.movedim(-1, 1)).movedim(1, -1)
        self.psp_post = self.spk.v_mem_recorded.movedim(1, -1)
        self.reset()

        return out

    def reset(self):
        self.spk.reset_states()


num_timesteps = 100
tau = 10
thr = 1
width_grad = 1
scale_grad = 1


for j in range(10):
    slyr = SlayerNet(num_timesteps, tau, thr, width_grad, scale_grad).cuda()
    exod = ExodusNet(num_timesteps, tau, thr, width_grad, scale_grad).cuda()
    exod.lin.weight.data = slyr.lin.weight.detach()

    for i in range(4):
        inp = torch.rand((1, 2, 4, 4, num_timesteps)).cuda()
        out_s = slyr(inp)
        out_e = exod(inp)

        print("Num spikes:", out_s.sum().item())
        assert torch.allclose(exod.psp_post, slyr.psp_post, atol=5e-3, rtol=1e-6)
        assert torch.allclose(out_s, out_e)
