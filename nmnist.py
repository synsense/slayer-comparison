import os
import torch
import tonic
import pytorch_lightning as pl
import numpy as np
from tonic import datasets, transforms, CachedDataset
from torch.utils.data import DataLoader, Subset


class NMNIST(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        first_saccade_only=False,
        dt=None,
        n_time_bins=None,
        num_workers=4,
        download_dir="./data",
        cache_dir="./cache/NMNIST/",
        fraction=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transforms.ToFrame(
            sensor_size=datasets.NMNIST.sensor_size,
            time_window=dt,
            n_time_bins=n_time_bins,
        )

    def prepare_data(self):
        datasets.NMNIST(self.hparams.download_dir, train=True)
        datasets.NMNIST(self.hparams.download_dir, train=False)

    def setup(self, stage=None):
        trainset = datasets.NMNIST(
            self.hparams.download_dir,
            train=True,
            transform=self.transform,
            first_saccade_only=self.hparams.first_saccade_only,
        )
        trainset = CachedDataset(
            trainset, cache_path=os.path.join(self.hparams.cache_dir, "train")
        )
        self.train_data = Subset(
            trainset, indices=torch.arange(len(trainset))[::int(1/self.hparams.fraction)]
        )

        validset = datasets.NMNIST(
            self.hparams.download_dir,
            train=False,
            transform=self.transform,
            first_saccade_only=self.hparams.first_saccade_only,
        )
        validset = CachedDataset(
            validset, cache_path=os.path.join(self.hparams.cache_dir, "test")
        )
        self.valid_data = Subset(
            validset, indices=torch.arange(len(validset))[::int(1/self.hparams.fraction)]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=True),
            shuffle=True,
            prefetch_factor=4,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=True),
            prefetch_factor=4,
            drop_last=True,
        )
