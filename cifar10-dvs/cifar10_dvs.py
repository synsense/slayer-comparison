import os

import pytorch_lightning as pl
import tonic
import torch
import torchvision
from tonic import DiskCachedDataset, datasets, transforms
from torch.utils.data import DataLoader, Subset


class CIFAR10DVS(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        n_time_bins,
        spatial_factor=1.0,
        num_workers=6,
        download_dir="./data",
        cache_dir="./cache/CIFAR10DVS/",
        augmentation=False,
        **kw_args
    ):
        super().__init__()
        self.save_hyperparameters()
        sensor_size = list(datasets.CIFAR10DVS.sensor_size)
        sensor_size[0] = int(sensor_size[0] * spatial_factor)
        sensor_size[1] = int(sensor_size[1] * spatial_factor)
        self.transform = torchvision.transforms.Compose(
            [
                transforms.Downsample(time_factor=1.0, spatial_factor=spatial_factor),
                transforms.ToFrame(
                    sensor_size=sensor_size,
                    n_time_bins=n_time_bins,
                    include_incomplete=True,
                ),
            ]
        )

        aug_deg = 25
        aug_shift = 0.15
        self.augmentation = (
            torchvision.transforms.Compose(
                [
                    torch.from_numpy,
                    torchvision.transforms.RandomAffine(
                        degrees=aug_deg,
                        translate=(aug_shift, aug_shift),
                        scale=(0.9, 1.1),
                    ),
                ]
            )
            if augmentation
            else None
        )

    def prepare_data(self):
        datasets.CIFAR10DVS(self.hparams.download_dir)

    def setup(self, stage=None, reset_cache=False):
        dataset = datasets.CIFAR10DVS(
            self.hparams.download_dir, transform=self.transform
        )
        train_idx = torch.arange(10000).reshape(10, -1)[:, :800].flatten()
        test_idx = torch.arange(10000).reshape(10, -1)[:, 800:].flatten()

        trainset = Subset(dataset, train_idx)
        validset = Subset(dataset, test_idx)

        self.train_data = DiskCachedDataset(
            dataset=trainset,
            cache_path=os.path.join(self.hparams.cache_dir, "train"),
            transform=self.augmentation,
            reset_cache=reset_cache,
        )

        self.valid_data = DiskCachedDataset(
            dataset=validset,
            cache_path=os.path.join(self.hparams.cache_dir, "test"),
            reset_cache=reset_cache,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=True),
            shuffle=True,
            prefetch_factor=4,
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

    def test_dataloader(self):
        return self.val_dataloader()
