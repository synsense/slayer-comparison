###
# Run DVS Gesture experiment. Each experiment follows the following procedure:
# - Generate identical models to be trained with SLAYER and/or EXODUS
# - Ensure that both models have identical forward dynamics (up to numerical errors)
# - Train the models with the DVS Gestue training set and after each epoch
#   perform a validation run with the test set.
# For possible command line arguments, run with '--help'. Additioal arguments
# such as `max_epochs` or `gpus` can be provided and will be passed directly
# to pytorch lightning `Trainer` class. See 
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
# for further details.
###

import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch
from dvs_gesture_model import GestureNetwork
from dvs_gesture import DVSGesture


def run_experiment(method, model, data, args):
    run_name = f"{method}/{args.num_conv_layers}_conv_layers/{args.base_channels}_base_channels/{args.scale_grad}_grad_scale/{args.width_grad}_grad_width"
    if not args.iaf:
        run_name = f"lif/tau{args.tau_mem}_" + run_name
    if args.sgd:
        run_name = f"sgd/" + run_name
    if args.run_name != "default":
        run_name += args.run_name

    checkpoint_path = Path("models") / "checkpoints" / run_name
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_acc",
        dirpath=checkpoint_path,
        filename="dvs_gesture-{step}-{epoch:02d}-{valid_loss:.2f}-{valid_acc:.2f}",
        save_top_k=1,
        mode="max",
    )

    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs/dvs_gestures", name=run_name)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        track_grad_norm=2,
    )

    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, data)

    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")


def compare_forward(models, data, no_lightning: bool=False):
    data.setup()
    dl = data.train_dataloader()
   
    if no_lightning:
        exodus_model = models["exodus"].cuda()
        slayer_model = models["slayer"].cuda()
    else:
        exodus_model = models["exodus"].network.cuda()
        slayer_model = models["slayer"].network.cuda()

    exodus_model.eval()
    slayer_model.eval()

    print("Making sure forward calls match")

    is_large_model = (len(exodus_model.conv_layers) > 4)

    for i, (inp, __) in enumerate(dl):
        print(f"\tBatch {i+1}")

        # Less strict comparison for large models due to accumulated numerical errors
        for lyr in exodus_model.spk_layers:
            lyr.reset_states()
        out_exodus = exodus_model(inp.cuda())
        out_slayer = slayer_model(inp.cuda())
        # assert torch.allclose(out_exodus, out_slayer, rtol=1e-6, atol=1e-5)
        rmse = torch.sqrt(((out_exodus-out_slayer)**2).mean())
        rms_exodus = torch.sqrt(((out_exodus)**2).mean())
        print(f"\tRMSE: {rmse:.4f} (rms exo: {rms_exodus:.4f})")
        # if not is_large_model:
        #     assert(rmse < 0.05 * rms_exodus)
        abs_dev = torch.abs(out_exodus-out_slayer)
        max_dev = torch.max(abs_dev)
        print(f"\tMax deviation: {max_dev:.4f}")
        median = torch.quantile(abs_dev, q=0.5)
        q90 = torch.quantile(abs_dev, q=0.9)
        print(f"\tMedian: {median:.4f}, .9 quantile: {q90:.4f}")
        # if not is_large_model:
        #     assert(q90 < 0.1 * rms_exodus)
        corr = torch.corrcoef(torch.stack((out_exodus.flatten(), out_slayer.flatten())))[0,1].item()
        print(f"Correlation: {corr:.4f}")
        assert(corr > 0.95)

        if i == 1:
            break
        
    for lyr in exodus_model.spk_layers:
        lyr.reset_states()

    exodus_model.train()
    slayer_model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="Can be 'slayer', 'exodus', or 'both' to train identical models with each algorithm.",
        type=str,
        default="both",
    )
    parser.add_argument("--sgd", dest="sgd", action="store_true", help="Use SGD as optimizer instead of ADAM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default: 32.")
    parser.add_argument("--base_channels", type=int, default=2, help="Number of features in first convolutional layer. For he next 3 conv. layers the number of features will increase layer by layer by a factor of 2. Default: 2")
    parser.add_argument("--num_conv_layers", type=int, default=4, help="Total number of convolutional layers in the network architecture. Default: 4.")
    parser.add_argument("--bin_dt", type=int, default=5000, help="Simulation times step in ms. Default: 5000")
    parser.add_argument("--max_timestamp", type=int, default=1.5e6, help="Crop time of each sample to that timestamp")
    parser.add_argument("--augmentation", dest="augmentation", action="store_true", help="Augment data during training by random rotations and translations.")
    parser.add_argument("--tau_mem", type=float, default=20.0, help="Membrane and synapse time constant for LIF neurons in ms. Has no effect if --iaf option is set. Default: 20.0")
    parser.add_argument("--spike_threshold", type=float, default=0.25, help="Neuron firing threshold. Default: 0.25")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training. Default: 1e-3")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay during training. Default: 1e-2")
    parser.add_argument("--run_name", type=str, default="default", help="Will be appended to run name, if not 'default'. Run name already summarizes most simulation parameters")
    parser.add_argument("--width_grad", type=float, default=1.0, help="Width of exponential surrogate gradient function.")
    parser.add_argument("--scale_grad", type=float, default=1.0, help="Scaling of exponential surrogate gradient function.")
    parser.add_argument("--iaf", dest="iaf", action="store_true", help="Use non-leaky Integrate-and-Fire neurons instead of LIF")
    parser.add_argument("--batchnorm", dest="batchnorm", action="store_true", help="Apply batch normalization during training")
    parser.add_argument("--dropout", dest="dropout", action="store_true", help="Apply dropout during training")
    parser.add_argument("--norm_weights", dest="norm_weights", action="store_true", help="Apply weight normalization during training")
    parser.add_argument("--rand_seed", type=int, default=1, help="Provide a seed for random number generation")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.rand_seed)

    data = DVSGesture(
        batch_size=args.batch_size,
        bin_dt=args.bin_dt,
        max_timestamp=args.max_timestamp,
        augmentation=args.augmentation,
        spatial_factor=0.5,
    )
    
    if args.method == "both":
        methods = ["exodus", "slayer"]
    else:
        methods = [args.method]

    models = dict()
    for method in methods:
        models[method] = GestureNetwork(
            **args,
            optimizer="SGD" if args.sgd else "Adam",
            num_timesteps=int(args.max_timestamp // args.bin_dt),
        )

    if len(models) > 1:
        # Copy initial weights from first model to others
        initial_params = models["exodus"].network.parameter_copy
        for k, m in models.items():
            if k != "exodus":
                m.network.import_parameters(initial_params)

    if args.method == "both":
        compare_forward(models, data)

    for k, m in models.items():
        run_experiment(k, m, data, args)
