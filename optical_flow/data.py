"""
Dataset classes for motion toy tasks
"""
from typing import Optional, Union, Callable, Dict, Tuple, Iterable, List
from pathlib import Path
from os import path, walk, listdir
import h5py

import torch
import torchvision.transforms
import numpy as np
import torch.nn.functional as F

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

class InvertDirDataset(torch.utils.data.Dataset):
    """
    Dataset class that generates samples from a given raster of events. Samples
    with higher indices are generated by inverting original raster along time
    axis. Samples are contiguous subsets of raster of given size and shifted by
    given step_size:
        ``sample = data[..., idx * step_size: idx * step_size + sample_size]``
    where `data` is either the raster or its inversion.

    Parameters
    ----------
    raster: np.array
        Event raster. Dimensions: Polarity-Y-X-Time
    sample_size: int
        Number of time steps per sample
    step_size: int
        By how many timesteps are samples shifted -> samples will overlap if
        `step_size` < `sample_size`
    downsample: Optional[int]
        If not `None`, collapse `downsample` subsequent timesteps to one
    """

    def __init__(self, raster, sample_size, step_size, downsample=None):
        self.raster = torch.from_numpy(raster).float()

        if downsample is not None:
            # Combine neighboring frames to one
            self.raster = torch.nn.functional.avg_pool2d(self.raster, (1, downsample))
            self.raster *= downsample
        # Flip along time axis
        self.raster_flipped = self.raster.flip(-1)
        self.sample_size = sample_size
        self.step_size = step_size
        # Number of samples per direction
        self.size = (self.raster.shape[-1] - sample_size) // step_size + 1
        self.start_idcs = np.arange(self.size) * step_size

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int
            Sample index, must be in [0, 2 * self.size)

        Returns
        -------
        torch.tensor
            Event raster. Dimensions: Polarity-Y-X-Time (Shape: 2-Y-X-self.sample_size)
        int
            Target class - 1 for time-inverted, 0 else
        torch.tensor
            One-hot encoding of target class - Shape: (2, 1, 1, 1)
        """

        start_idx = self.start_idcs[idx % self.size]
        slc = slice(start_idx, start_idx + self.sample_size)
        if idx >= 2 * self.size:
            raise IndexError(f"Index must be between 0 and {2 * self.size - 1}")
        if idx >= self.size:
            # Inverted direction
            return self.raster_flipped[..., slc], 1, torch.tensor([[[[0]]], [[[1]]]])
        else:
            # Original direction
            return self.raster[..., slc], 0, torch.tensor([[[[1]]], [[[0]]]])

    def __len__(self):
        return 2 * self.size