import argparse
import pytorch_lightning as pl
from nmnist_exodus import SinabsNetwork
from nmnist import NMNIST
import torch
import numpy as np


if __name__ == "__main__":
    pl.seed_everything(123)
    np.random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", help="Can be 'sinabs' or 'exodus'.", type=str, default="exodus"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--first_saccade_only", dest="first_saccade_only", action="store_true"
    )
    parser.add_argument("--n_time_bins", type=int, default=300)
    parser.add_argument("--tau_mem", type=float, default=20.0)
    parser.add_argument("--spike_threshold", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.set_defaults(first_saccade_only=False)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = SinabsNetwork(
        batch_size=args.batch_size,
        tau_mem=args.tau_mem,
        spike_threshold=args.spike_threshold,
        learning_rate=args.learning_rate,
        method=args.method,
    )

    data = NMNIST(
        batch_size=args.batch_size,
        first_saccade_only=args.first_saccade_only,
        n_time_bins=args.n_time_bins,
        num_workers=4,
        download_dir="./data",
        cache_dir="./cache/NMNIST/",
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath="models/checkpoints",
        filename="nmnist-{step}-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=True,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, data)
