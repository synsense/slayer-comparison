import torch, math
import pytorch_lightning as pl
from tonic import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import tonic


class ToRaster():
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim

    def __call__(self, events):
        # tensor has dimensions (time_steps, encoding_dim)
        tensor = np.zeros((events["t"].max()+1, self.encoding_dim), dtype=int)
        np.add.at(tensor, (events["t"], events["x"]), 1)
        return tensor[:250,:]


class SSC(pl.LightningDataModule):
    def __init__(self, batch_size, encoding_dim=700, dt=4000, num_workers=6, download_dir='./data'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download_dir = download_dir

        self.transform = transforms.Compose([
            transforms.Downsample(time_factor=1/dt, spatial_factor=encoding_dim/700),
            ToRaster(encoding_dim),
        ])
  
    def prepare_data(self):
        datasets.SSC(self.download_dir, split='train')
        datasets.SSC(self.download_dir, split='valid')
        datasets.SSC(self.download_dir, split='test')
  
    def setup(self, stage=None):
        self.train_data = datasets.SSC(self.download_dir, split='train', transform=self.transform)
        self.valid_data = datasets.SSC(self.download_dir, split='valid', transform=self.transform)
        self.test_data = datasets.SSC(self.download_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batch_size, 
                          collate_fn=tonic.collation.PadTensors(batch_first=True), shuffle=True)
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, num_workers=self.num_workers, batch_size=self.batch_size, 
                          collate_fn=tonic.collation.PadTensors(batch_first=True))

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.batch_size, 
                          collate_fn=tonic.collation.PadTensors(batch_first=True))
