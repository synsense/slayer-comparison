import argparse
import pytorch_lightning as pl
from poisson_exodus import SinabsNetwork
import torch


class PoissonSpike:
    def __init__(self, encoding_dim, n_time_steps):
        self.sample = torch.poisson(torch.rand((n_time_steps, encoding_dim)))
        self.target = torch.zeros((n_time_steps))
        self.target[20] = 1.
        self.target[35] = 1.
        self.target[45] = 1.
        self.target = torch.poisson(torch.rand((n_time_steps)))
    
    def __getitem__(self, index):
        return (self.sample, self.target)

    def __len__(self):
        return 1

if __name__ == "__main__":
    pl.seed_everything(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="Can be 'sinabs' or 'exodus'.", type=str, default="exodus")
    parser.add_argument("--encoding_dim", type=int, default=250)
    parser.add_argument("--hidden_dim", type=int, default=25)
    parser.add_argument("--tau_mem", type=float, default=10.0)
    parser.add_argument("--spike_threshold", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--n_time_steps", type=int, default=50)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = SinabsNetwork(
        tau_mem=args.tau_mem,
        spike_threshold=args.spike_threshold,
        learning_rate=args.learning_rate,
        method=args.method,
        encoding_dim=args.encoding_dim,
        hidden_dim=args.hidden_dim,
    )

    data = torch.utils.data.DataLoader(PoissonSpike(encoding_dim=args.encoding_dim, n_time_steps=args.n_time_steps))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath="models/checkpoints",
        filename="poisson-{step}-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=True,
        # callbacks=[checkpoint_callback],
        # log_every_n_steps=10,
    )

    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, data)

    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
