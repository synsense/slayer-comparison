####
# Lie simple_lif.py, but with alpha kernels (i.e. synaptic currents)
####

import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/..")

import torch

from smnist_slayer import SlayerNetwork
from smnist_exodus import ExodusNetwork


class ExodusNet(ExodusNetwork):
    def forward(self, x):
        self.reset_states()
        return super().forward(x)


n_time_bins = 784
batch_size = 2
tau_mem = 20.0
spike_threshold = 0.1
width_grad = 1
scale_grad = 1
learning_rate = 1e-3
encoding_dim = 80
hidden_dim1 = 128
hidden_dim2 = 256
init_weights_path = "../init_weights_smnist.pt"
decoding_func = "sum_loss"


def transfer_dense_weight(slayer_layer, exo_layer):
    # Slayer conv weights have additional dimension for time (with size 1)
    slayer_layer.weight.data = exo_layer.weight.detach()
    slayer_layer.weight_g.data = exo_layer.weight_g.detach()
    slayer_layer.weight_v.data = exo_layer.weight_v.detach()


def transfer_weights(src_slr, tgt_exo):
    transfer_dense_weight(tgt_exo.network[0], src_slr.linear1)
    transfer_dense_weight(tgt_exo.network[2], src_slr.linear2)
    transfer_dense_weight(tgt_exo.network[4], src_slr.linear3)


cont = True
# Try different inputs and network instances
for j in range(10):
    if cont:
        slyr = SlayerNetwork(
            n_time_bins=n_time_bins,
            encoding_dim=encoding_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            tau_mem=tau_mem,
            spike_threshold=spike_threshold,
            learning_rate=learning_rate,
            width_grad=width_grad,
            scale_grad=scale_grad,
            init_weights_path=init_weights_path,
            decoding_func=decoding_func,
        ).cuda()
        exod = ExodusNet(
            encoding_dim=encoding_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            tau_mem=tau_mem,
            spike_threshold=spike_threshold,
            learning_rate=learning_rate,
            width_grad=width_grad,
            scale_grad=scale_grad,
            init_weights_path=init_weights_path,
            decoding_func=decoding_func,
        ).cuda()

        # transfer_weights(slyr, exod)

        for i in range(4):
            inp = torch.rand((batch_size, n_time_bins, encoding_dim)).cuda()
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
                # cont = False  # Break outer loop

                # out_exo = []
                # x = inp.clone()
                # for lyr in exod.network:
                #     x = lyr(x)
                #     out_exo.append(x)
                # unflat = torch.nn.Unflatten(0, (batch_size, -1))
                # out_exo_reshaped = [unflat(x).movedim(1, -1) for x in out_exo[1:-1]]

                # out_slayer = []
                # x = inp.clone().movedim(1, -1)
                # x = slyr.conv1(x)
                # out_slayer.append(x)
                # x = slyr.slayer.spike(slyr.slayer.psp(x))
                # out_slayer.append(x)
                # x = slyr.pool1(x)
                # out_slayer.append(x)
                # x = slyr.conv2(x)
                # out_slayer.append(x)
                # x = slyr.slayer.spike(slyr.slayer.psp(x))
                # out_slayer.append(x)
                # x = slyr.pool2(x)
                # out_slayer.append(x)
                # if architecture == "larger":
                #     x = slyr.conv3(x)
                #     out_slayer.append(x)
                #     x = slyr.slayer.spike(slyr.slayer.psp(x))
                #     out_slayer.append(x)
                # x = slyr.fc1(x)
                # out_slayer.append(x)
                # if architecture == "larger":
                #     x = slyr.slayer.spike(slyr.slayer.psp(x))
                #     out_slayer.append(x)
                #     x = slyr.fc2(x)
                #     out_slayer.append(x)

                break
