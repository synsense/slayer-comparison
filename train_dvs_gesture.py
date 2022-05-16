import argparse
from pathlib import Path
from time import strftime
import pytorch_lightning as pl
import torch
from dvs_gesture_model import GestureNetwork
from data_modules.dvs_gesture import DVSGesture

def run_experiment(method, model, data, args):
    timestamp = strftime("%Y_%m_%d_%H_%M_%S") 
    run_name = f"{method}_{args.num_conv_layers}lyrs_s{args.scale_grad}_w{args.width_grad}_{timestamp}"
    if args.run_name != "default":
        run_name += args.run_name

    checkpoint_path = Path("models") / "checkpoints" / run_name
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        dirpath=checkpoint_path,
        filename="dvs_gesture-{step}-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs/dvs_gestures", name=run_name)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        accelerator="gpu",
        track_grad_norm=2,
    )

    trainer.logger.log_hyperparams(model.hparams)
    trainer.fit(model, data)

    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")


def generate_models(args):
    if args.method == "both":
        methods = ["exodus", "slayer"]
    else:
        methods = [args.method]

    models = dict()
    for method in methods:
        models[method] = GestureNetwork(
            batch_size=args.batch_size,
            tau_mem=args.tau_mem,
            spike_threshold=args.spike_threshold,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            width_grad=args.width_grad,
            scale_grad=args.scale_grad,
            init_weights_path=args.init_weight_path,
            iaf=args.iaf,
            base_channels=args.base_channels,
            num_conv_layers=args.num_conv_layers,
            method=method,
            num_timesteps=args.num_time_bins,
            optimizer="SGD" if args.sgd else "Adam",
            batchnorm=args.batchnorm,
            dropout=args.dropout,
            weight_norm=args.weight_norm,
        )

    if len(models) > 1:
        # Copy initial weights from first model to others
        initial_params = models["exodus"].network.parameter_copy
        for k, m in models.items():
            if k != "exodus":
                m.network.import_parameters(initial_params)

    return models

def compare_forward(models, data):
    data.setup()
    dl = data.train_dataloader()
    
    exodus_model = models["exodus"].network.cuda()
    slayer_model = models["slayer"].network.cuda()

    exodus_model.eval()
    slayer_model.eval()

    print("Making sure forward calls match")

    large_model = (len(exodus_model.conv_layers) > 4)

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
        # if not large_model:
        #     assert(rmse < 0.05 * rms_exodus)
        abs_dev = torch.abs(out_exodus-out_slayer)
        max_dev = torch.max(abs_dev)
        print(f"\tMax deviation: {max_dev:.4f}")
        median = torch.quantile(abs_dev, q=0.5)
        q90 = torch.quantile(abs_dev, q=0.9)
        print(f"\tMedian: {median:.4f}, .9 quantile: {q90:.4f}")
        # if not large_model:
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
    pl.seed_everything(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="Can be 'slayer' or 'exodus'.",
        type=str,
        default="exodus",
    )
    parser.add_argument("--sgd", dest="sgd", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=2)
    parser.add_argument("--num_conv_layers", type=int, default=4)
    parser.add_argument("--bin_dt", type=int, default=5000)
    parser.add_argument("--dataset_fraction", type=float, default=1.0)
    parser.add_argument("--spatial_factor", type=float, default=1.0)
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--tau_mem", type=float, default=20.0)
    parser.add_argument("--spike_threshold", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--width_grad", type=float, default=1.0)
    parser.add_argument("--scale_grad", type=float, default=1.0)
    parser.add_argument("--init_weight_path", type=str, default=None)
    parser.add_argument("--iaf", dest="iaf", action="store_true")
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--num_time_bins", type=int, default=300)
    parser.add_argument("--batchnorm", dest="batchnorm", action="store_true")
    parser.add_argument("--dropout", dest="dropout", action="store_true")
    parser.add_argument("--weight_norm", dest="weight_norm", action="store_true")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    data = DVSGesture(
        batch_size=args.batch_size,
        bin_dt=args.bin_dt,
        fraction=args.dataset_fraction,
        augmentation=args.augmentation,
        spatial_factor=args.spatial_factor,
        num_time_bins=args.num_time_bins,
    )
    
    for i_run in range(args.num_repetitions):
       
        models = generate_models(args)

        if args.method == "both":
            compare_forward(models, data)

        for k, m in models.items():
            run_experiment(k, m, data, args)
