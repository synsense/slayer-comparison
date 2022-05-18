import argparse
import pytorch_lightning as pl
from ssc_exodus import ExodusNetwork
from ssc_slayer import SlayerNetwork
from ssc import SSC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand_seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--encoding_dim", type=int, default=100)
    parser.add_argument(
        "--decoding_func",
        help="Use 'sum_loss', 'max_over_time'  or 'last_ts'.",
        type=str,
        default="max_over_time",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--tau_mem", type=float, default=30.0)
    parser.add_argument("--tau_syn", type=float, default=None)
    parser.add_argument("--spike_threshold", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--width_grad", type=float, default=1.0)
    parser.add_argument("--scale_grad", type=float, default=1.0)
    parser.add_argument("--n_hidden_layers", type=int, default=2)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.rand_seed)

    data = SSC(
        batch_size=args.batch_size,
        encoding_dim=args.encoding_dim,
        num_workers=4,
        download_dir="./data",
    )

    slayer_model = SlayerNetwork(**dict_args, n_time_bins=250, output_dim=35)
    init_weights = slayer_model.state_dict()

    exodus_model = ExodusNetwork(**dict_args, init_weights=init_weights, output_dim=35)

    checkpoint_path = "models/checkpoints"

    for run_name, model in [['ssc-slayer', slayer_model], ['ssc-exodus', exodus_model]]:
        run_name += f"-{args.tau_mem}-{args.scale_grad}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="valid_loss",
            dirpath=checkpoint_path + '/' + run_name,
            filename="{run_name}-{step}-{epoch:02d}-{valid_loss:.2f}-{val_acc:.2f}",
            save_top_k=1,
            mode="min",
        )

        logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
        trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=20
        )

        trainer.logger.log_hyperparams(model.hparams)
        trainer.fit(model, data)

        print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
