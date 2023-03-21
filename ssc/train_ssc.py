import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from ssc_exodus import ExodusNetwork
from ssc_slayer import SlayerNetwork
from ssc_jelly import JellyNetwork
from ssc import SSC

class LogGrads(pl.callbacks.Callback):
    def on_after_backward(self, trainer, pl_module):
        grad_dict = {k: p.grad for k, p in pl_module.named_parameters() if p.grad is not None}
        save_dir = Path(trainer.log_dir) / "grads.pt"
        torch.save(grad_dict, save_dir)
        for k, v in grad_dict.items():
            print(k, v.sum().item(), v.mean().item(), v.std().item())
        print(f"Gradients have been saved to {save_dir}")

    def on_before_backward(self, trainer, pl_module, loss):
        print("Loss:", loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand_seed", type=int, default=123, help="Provide a seed for random number generation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size. Default: 128")
    parser.add_argument("--encoding_dim", type=int, default=100, help="Number of neurons in encoding layer. Default: 100")
    parser.add_argument(
        "--decoding_func",
        help="Use 'sum_loss', 'max_over_time'  or 'last_ts'.",
        type=str,
        default="max_over_time",
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Number of neurons in hidden layer(s). Default: 128")
    parser.add_argument("--tau_mem", type=float, default=30.0, help="Membrane time constant in ms")
    parser.add_argument("--spike_threshold", type=float, default=1.0, help="Neuron firing threshold. Default: 1")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate during training. Default: 1e-3")
    parser.add_argument("--width_grad", type=float, default=1.0, help="Width of exponential surrogate gradient function. Default: 1.0")
    parser.add_argument("--scale_grad", type=float, default=1.0, help="Scaling of exponential surrogate gradient function. Default: 1.0")
    parser.add_argument("--n_hidden_layers", type=int, default=2, help="Number of hidden layers. Default: 2")
    parser.add_argument("--optimizer", type=str, default="adam", help="Define to use 'adam' or 'sgd'. Default: adam")
    parser.add_argument("--grad_mode", action="store_true", help="Only run one iteration for each model and store gradients for later comparison")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.rand_seed)

    data = SSC(
        batch_size=args.batch_size,
        encoding_dim=args.encoding_dim,
        num_workers=4,
        download_dir="./data",
        shuffle=(not args.grad_mode)
    )

    slayer_model = SlayerNetwork(**dict_args, n_time_bins=250, output_dim=35)
    init_weights = slayer_model.state_dict()

    exodus_model = ExodusNetwork(**dict_args, init_weights=init_weights, output_dim=35)
    jelly_model = JellyNetwork(**dict_args, init_weights=init_weights, output_dim=35)

    model_dict = {"slayer": slayer_model, "exodus": exodus_model, "jelly": jelly_model}
    checkpoint_path = "models/checkpoints"

    if args.grad_mode:
        sinabs_model = ExodusNetwork(**dict(dict_args, backend="sinabs"), init_weights=init_weights, output_dim=35)
        model_dict["bptt"] = sinabs_model

    for model_name, model in model_dict.items():
        run_name = f"ssc/{model_name}/{args.tau_mem}/{args.scale_grad}/{args.optimizer}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="valid_acc",
            dirpath=checkpoint_path + "/" + run_name,
            filename="{run_name}-{step}-{epoch:02d}-{valid_loss:.2f}-{valid_acc:.2f}",
            save_top_k=1,
            mode="max",
        )
        
        if args.grad_mode:
            callbacks = [LogGrads()]
            args.num_sanity_val_steps = 0
            args.max_steps = 1
        else:
            callbacks = [checkpoint_callback]
        logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
        trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, callbacks=callbacks, log_every_n_steps=20
        )

        trainer.logger.log_hyperparams(model.hparams)
        trainer.fit(model, data)

        print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
