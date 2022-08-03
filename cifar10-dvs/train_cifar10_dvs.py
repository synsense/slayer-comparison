import argparse
from pathlib import Path

import pytorch_lightning as pl

from cifar10_dvs import CIFAR10DVS
from exodus_model import ExodusNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--tau_mem", type=float)
    parser.add_argument("--n_time_bins", type=int)
    parser.add_argument("--rand_seed", type=int, default=0)
    parser.add_argument("--spatial_factor", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--spike_threshold", type=float, default=1.)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--lr_scheduler_t_max", type=int, default=0)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.rand_seed)

    data = CIFAR10DVS(**dict_args)

    model = ExodusNetwork(
        tau_mem=args.tau_mem,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        spike_threshold=args.spike_threshold,
        optimizer=args.optimizer,
        lr_scheduler_t_max=args.lr_scheduler_t_max
    )

    run_name = f"exodus/{args.optimizer}/{args.tau_mem}_tau_mem/{args.n_time_bins}_time_bins"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="accuracy/validation",
        dirpath=Path("models") / "checkpoints" / run_name,
        filename="{run_name}-step={step}-epoch={epoch:02d}-valid_acc={accuracy/validation:.2f}-valid_loss={loss/validation:.2f}",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs/cifar10_dvs", name=run_name
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )

    trainer.fit(model, data)
