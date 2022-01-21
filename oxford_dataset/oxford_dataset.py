import torch
from torch.utils.data import Dataset
import lava.lib.dl.slayer as slayer


class OxfordDataset(Dataset):
    def __init__(self):
        super(OxfordDataset, self).__init__()
        self.input  = slayer.io.read_1d_spikes('data/input.bs1' )
        self.target = slayer.io.read_1d_spikes('data/output.bs1')
        self.target.t = self.target.t.astype(int)

    def __getitem__(self, _):
        return (
            self.input.fill_tensor(torch.zeros(1, 1, 200, 2000)).squeeze(),  # input
            self.target.fill_tensor(torch.zeros(1, 1, 200, 2000)).squeeze(), # target
        )

    def __len__(self):
        return 1 # just one sample for this problem