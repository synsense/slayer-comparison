import argparse
import pytorch_lightning as pl
from smnist_exodus import ExodusNetwork
from smnist_slayer import SlayerNetwork
from smnist import SMNIST


if __name__ == "__main__":
    pl.seed_everything(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", help="Use 'slayer' or 'exodus'.", type=str, default="exodus"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--encoding_dim", type=int, default=80)
    parser.add_argument(
        "--encoding_func", help="Use 'poisson' or 'rbf'.", type=str, default="rbf"
    )
    parser.add_argument(
        "--decoding_func",
        help="Use 'sum_loss' or 'last_ts'.",
        type=str,
        default="sum_loss",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--tau_mem", type=float, default=20.0)
    parser.add_argument("--tau_syn", type=float, default=10.0)
    parser.add_argument("--spike_threshold", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--width_grad", type=float, default=1.0)
    parser.add_argument("--scale_grad", type=float, default=1.0)
    parser.add_argument("--init_weights_path", type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    if args.method == "exodus":
        model = ExodusNetwork(**dict_args)
    elif args.method == "slayer":
        model = SlayerNetwork(**dict_args, n_time_bins=784)
    else:
        raise ValueError(f"Method {args.method} not recognized.")

    data = SMNIST(
        batch_size=args.batch_size,
        encoding_dim=args.encoding_dim,
        encoding_func=args.encoding_func,
        num_workers=4,
        download_dir="./data",
    )

    checkpoint_path = "models/checkpoints"
    run_name = args.method
    if args.run_name != "default":
        run_name += "/" + args.run_name
        checkpoint_path += "/" + args.run_name

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="smnist-{step}-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=10
    )

    trainer.fit(model, data)

    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
