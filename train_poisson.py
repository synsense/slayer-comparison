import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import sinabs.activation as sa
from sinabs.slayer.layers import LIF, ExpLeakSqueeze
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_context("paper")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding_dim", type=int, default=250)
    parser.add_argument("--hidden_dim", type=int, default=25)
    parser.add_argument("--tau_mem", type=float, default=10.0)
    parser.add_argument("--spike_threshold", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--n_time_steps", type=int, default=500)
    args = parser.parse_args()

    act_fn = sa.ActivationFunction(
                spike_threshold=args.spike_threshold,
                spike_fn=sa.SingleSpike,
                reset_fn=sa.MembraneSubtract(),
                surrogate_grad_fn=sa.SingleExponential(),
            )

    model = nn.Sequential(
                nn.Linear(args.encoding_dim, args.hidden_dim, bias=False),
                LIF(tau_mem=args.tau_mem, activation_fn=act_fn),
                nn.Linear(args.hidden_dim, 1, bias=False),
                LIF(tau_mem=args.tau_mem, activation_fn=act_fn),
            ).cuda()

    torch.manual_seed(123)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    input_spikes = (torch.rand(1, args.n_time_steps, args.encoding_dim) > 0.95).float().cuda() * 1e8
    target = torch.zeros((1, args.n_time_steps, 1)).float().cuda()
    target[0, torch.randint(args.n_time_steps//5, args.n_time_steps, (4, )), 0] = 1

    out_spikes = []
    for epoch in range(200):
        optimiser.zero_grad()
        model[1].zero_grad()
        model[3].zero_grad()
        out = model(input_spikes)
        loss = criterion(out, target)
        loss.backward()
        optimiser.step()
        print(f"loss: {loss.item()}, n_spikes: {out.sum()}")

        out_spikes.append(np.ravel(out.detach().cpu().int().numpy()))

    input_spikes = input_spikes.detach().cpu().int().squeeze(1).numpy()
    output_spikes = np.array(out_spikes).T
    target_spikes = target.detach().cpu().int().squeeze(1).numpy()

    plt.eventplot(output_spikes)
    plt.show()
    
    # fig = plt.figure(figsize=(4, 8))
    # ax = fig.add_subplot(111)
    # ax.scatter(np.where(out_spks)[0] * dt, np.where(out_spks)[1], s=1.)
    # for spk in np.where(tgt_spks)[0]:
    #     ax.axvspan(spk * dt - 0.1, spk*dt + 0.1, alpha=0.2, color='red')
    # ax.set_xlabel('Time (ms)')
    # ax.set_ylabel('Epoch')
    # ax.set_xlim([0, 50])
    # sns.despine()
    # plt.tight_layout()

    # plt.savefig('spike_pattern.png', dpi=200.)
    # plt.close('all')

