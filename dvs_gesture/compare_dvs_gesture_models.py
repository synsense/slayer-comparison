from dvs_gesture_model import SlayerNetwork, ExodusNetwork
import torch

kwargs_model = dict(
    batch_size=8,
    tau_mem=10,
    spike_threshold=0.1,
    base_channels=8,
    kernel_size=3,
    num_conv_layers=8,
    width_grad=1.0,
    scale_grad=1.0,
    iaf=False,
    num_timesteps=300,
    batchnorm=True,
    dropout=True,
)

exodus_model = ExodusNetwork(**kwargs_model).cuda()
slayer_model = SlayerNetwork(**kwargs_model).cuda()

slayer_model.import_parameters(exodus_model.parameter_copy)

slayer_model.eval()
exodus_model.eval()

if __name__ == "__main__":

    from data_modules.dvs_gesture import DVSGesture
   
    data = DVSGesture(
        batch_size=kwargs_model["batch_size"],
        bin_dt=5000,
        fraction=1.0,
        augmentation=False,
        spatial_factor=0.5,
    )
    data.setup()
    dl = data.val_dataloader()

    for i, (inp, __) in enumerate(dl):
        print(f"Batch {i+1}")

        for lyr in exodus_model.spk_layers:
            lyr.reset_states()
        out_exodus = exodus_model(inp.cuda())
        out_slayer = slayer_model(inp.cuda())
        # assert torch.allclose(out_exodus, out_slayer, rtol=1e-6, atol=1e-5)
        rmse = torch.sqrt(((out_exodus-out_slayer)**2).mean())
        rms_exodus = torch.sqrt(((out_exodus)**2).mean())
        print(f"RMSE: {rmse:.4f} (rms exo: {rms_exodus:.4f})")
        abs_dev = torch.abs(out_exodus-out_slayer)
        max_dev = torch.max(abs_dev)
        print(f"Max deviation: {max_dev:.4f}")
        median = torch.quantile(abs_dev, q=0.5)
        q90 = torch.quantile(abs_dev, q=0.9)
        print(f"Median: {median:.4f}, .9 quantile: {q90:.4f}")
        corr = torch.corrcoef(torch.stack((out_exodus.flatten(), out_slayer.flatten())))[0,1].item()
        print(f"Correlation: {corr:.4f}")
        if i == 2:
            break

        
