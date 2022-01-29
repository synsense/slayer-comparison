import torch, math
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils import data


class PoissonEncoder:
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim

    def __call__(self, x):
        x = x.ravel().repeat(self.encoding_dim, 1).T
        rand_matrix = torch.rand_like(x)
        spikes = (x > rand_matrix).float()
        return spikes


class RBFEncoder:
    def __init__(self, encoding_dim):
        assert encoding_dim >= 3
        self.encoding_dim = encoding_dim

    def _gaussian(self, x, mu=0.0, sigma=0.5):
        return (
            torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            / torch.sqrt(2 * torch.tensor(math.pi))
            / sigma
        )

    def __call__(self, x):
        # x: (1,28,28), 0-1
        x = x.ravel().repeat(self.encoding_dim, 1).T

        scale = 1.0 / (self.encoding_dim - 2)
        mus = torch.as_tensor(
            [(2 * i - 2) / 2 * scale for i in range(self.encoding_dim)]
        )
        sigmas = scale

        # for i in range(num_neurons):
        spikes = self._gaussian(x, mu=mus, sigma=sigmas)
        return spikes


class SMNIST(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        encoding_dim,
        encoding_func="poisson",
        num_workers=6,
        download_dir="./data",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download_dir = download_dir
        if encoding_func == "poisson":
            self.encoder = PoissonEncoder(encoding_dim=encoding_dim)
        elif encoding_func == "rbf":
            self.encoder = RBFEncoder(encoding_dim=encoding_dim)
        else:
            assert False

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                self.encoder,
            ]
        )

    def prepare_data(self):
        datasets.MNIST(self.download_dir, train=True, download=True)
        datasets.MNIST(self.download_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_data = datasets.MNIST(
            self.download_dir, train=True, download=False, transform=self.transform
        )
        self.test_data = datasets.MNIST(
            self.download_dir, train=False, download=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            drop_last=True,
        )
