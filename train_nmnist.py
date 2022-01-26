import argparse
import pytorch_lightning as pl
from nmnist_exodus import SinabsNetwork
from nmnist_slayer import SlayerNetwork
from nmnist import NMNIST


if __name__ == "__main__":
    pl.seed_everything(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="Can be 'slayer', 'sinabs' or 'exodus'.",
        type=str,
        default="exodus",
    )
    parser.add_argument(
        "--architecture", help="Can be 'paper' or 'larger'.", type=str, default="paper"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset_fraction", type=float, default=1.0)
    parser.add_argument(
        "--first_saccade_only", dest="first_saccade_only", action="store_true"
    )
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--n_time_bins", type=int, default=300)
    parser.add_argument("--tau_mem", type=float, default=20.0)
    parser.add_argument("--spike_threshold", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--width_grad", type=float, default=1.0)
    parser.add_argument("--scale_grad", type=float, default=1.0)
    parser.add_argument("--init_weight_path", type=str, default=None)
    parser.set_defaults(first_saccade_only=False, augmentation=False)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.method in ("sinabs", "exodus"):
        model = SinabsNetwork(
            batch_size=args.batch_size,
            tau_mem=args.tau_mem,
            spike_threshold=args.spike_threshold,
            learning_rate=args.learning_rate,
            method=args.method,
            architecture=args.architecture,
            width_grad=args.width_grad,
            scale_grad=args.scale_grad,
            init_weights_path=args.init_weight_path,
        )
    elif args.method == "slayer":
        model = SlayerNetwork(
            tau_mem=args.tau_mem,
            spike_threshold=args.spike_threshold,
            learning_rate=args.learning_rate,
            n_time_bins=args.n_time_bins,
            architecture=args.architecture,
            init_weights_path=args.init_weight_path,
        )
    else:
        raise ValueError(f"Method {args.method} not recognized.")

    data = NMNIST(
        batch_size=args.batch_size,
        first_saccade_only=args.first_saccade_only,
        n_time_bins=args.n_time_bins,
        num_workers=4,
        download_dir="./data",
        cache_dir="./cache/NMNIST/",
        fraction=args.dataset_fraction,
        augmentation=args.augmentation,
    )

    checkpoint_path = "models/checkpoints"
    run_name = args.method
    if args.run_name != "default":
        run_name += "/" + args.run_name
        checkpoint_path += "/" + args.run_name

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="nmnist-{step}-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=10
    )

    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, data)

    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
