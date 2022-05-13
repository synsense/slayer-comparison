import os
import torch
import tonic
import pytorch_lightning as pl
import torchvision
from tonic import datasets, transforms, DiskCachedDataset, MemoryCachedDataset, SlicedDataset
from torch.utils.data import DataLoader, Subset


class DVSGesture(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        bin_dt,
        spatial_factor=1.,
        num_workers=4,
        download_dir="./data",
        cache_dir="./cache/DVSGesture/",
        fraction=1,
        augmentation=False,
        num_time_bins=300,
    ):
        super().__init__()
        self.save_hyperparameters()
        sensor_size = list(datasets.DVSGesture.sensor_size)
        sensor_size[0] = int(sensor_size[0] * spatial_factor)
        sensor_size[1] = int(sensor_size[1] * spatial_factor)
        self.transform = torchvision.transforms.Compose([
            lambda events: events[events["t"] < num_time_bins * bin_dt],
            transforms.Downsample(time_factor = 1., spatial_factor=spatial_factor),
            transforms.ToFrame(
                sensor_size=sensor_size,
                time_window=bin_dt,
                include_incomplete=True,
            ),
        ])
            
        aug_deg = 10
        aug_shift = 0.05
        self.augmentation = (
            torchvision.transforms.Compose(
                [
                    torch.from_numpy,
                    torchvision.transforms.RandomAffine(
                        degrees=aug_deg, translate=(aug_shift, aug_shift)
                    ),
                ]
            )
            if augmentation
            else None
        )

    def prepare_data(self):
        datasets.DVSGesture(self.hparams.download_dir, train=True)
        datasets.DVSGesture(self.hparams.download_dir, train=False)

    def setup(self, stage=None):
        trainset = datasets.DVSGesture(
            self.hparams.download_dir,
            train=True,
            transform=self.transform,
        )
        trainset = DiskCachedDataset(
            dataset=trainset,
            cache_path=os.path.join(self.hparams.cache_dir, "train"),
            transform=self.augmentation,
            reset_cache=True,
        )
        self.train_data = Subset(
            trainset,
            indices=torch.arange(len(trainset))[:: int(1 / self.hparams.fraction)],
        )

        validset = datasets.DVSGesture(
            self.hparams.download_dir,
            train=False,
            transform=self.transform,
        )
        validset = DiskCachedDataset(
            dataset=validset,
            cache_path=os.path.join(self.hparams.cache_dir, "test"),
            reset_cache=True,
        )
        self.valid_data = Subset(
            validset,
            indices=torch.arange(len(validset))[:: int(1 / self.hparams.fraction)],
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
