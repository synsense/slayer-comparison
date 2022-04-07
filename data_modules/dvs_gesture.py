from curses import meta
import os
from importlib_metadata import metadata
import torch
import tonic
import pytorch_lightning as pl
import torchvision
from tonic import datasets, transforms, DiskCachedDataset, SlicedDataset
from torch.utils.data import DataLoader, Subset


class DVSGesture(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        bin_dt=1000,
        slice_dt=200000,
        num_workers=4,
        download_dir="./data",
        cache_dir="./cache/DVSGesture/",
        slice_metadata_dir="./cache/metadata/DVSGesture/",
        fraction=1,
        augmentation=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transform = transforms.ToFrame(
            sensor_size=datasets.DVSGesture.sensor_size,
            time_window=bin_dt,
        )
        aug_deg = 20
        aug_shift = 0.1
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
        )
        slicer = tonic.slicers.SliceByTime(time_window=self.hparams.slice_dt)
        trainset = SlicedDataset(
            dataset=trainset, 
            slicer=slicer, 
            metadata_path=os.path.join(
                self.hparams.slice_metadata_dir, 
                f"train/{self.hparams.slice_dt}"
            ),
            transform=self.transform,
        )
        trainset = DiskCachedDataset(
            dataset=trainset,
            cache_path=os.path.join(self.hparams.cache_dir, "train"),
            transform=self.augmentation,
        )
        self.train_data = Subset(
            trainset,
            indices=torch.arange(len(trainset))[:: int(1 / self.hparams.fraction)],
        )

        validset = datasets.DVSGesture(
            self.hparams.download_dir,
            train=False,
        )
        validset = SlicedDataset(
            dataset=validset, 
            slicer=slicer, 
            metadata_path=os.path.join(
                self.hparams.slice_metadata_dir, 
                f"test/{self.hparams.slice_dt}"
            ),
            transform=self.transform,
        )
        validset = DiskCachedDataset(
            dataset=validset,
            cache_path=os.path.join(self.hparams.cache_dir, "test"),
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

    def test_dataloader(self):
        return self.val_dataloader()
