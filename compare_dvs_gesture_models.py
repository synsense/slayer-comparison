from dvs_gesture_model import SlayerNetwork, ExodusNetwork
import torch

kwargs_model = dict(
    batch_size=8,
    tau_mem=10,
    spike_threshold=0.1,
    base_channels=8,
    kernel_size=3,
    num_conv_layers=4,
    width_grad=1.0,
    scale_grad=1.0,
    iaf=True,
    num_timesteps=300,
)

exodus_model = ExodusNetwork(**kwargs_model).cuda()
slayer_model = SlayerNetwork(**kwargs_model).cuda()

slayer_model.import_parameters(exodus_model.parameter_copy)

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
        out_exodus = exodus_model(inp.cuda())
        out_slayer = slayer_model(inp.cuda())
        # assert torch.allclose(out_exodus, out_slayer, rtol=1e-6, atol=1e-5)
        rmse = torch.sqrt(((out_exodus-out_slayer)**2).mean())
        rms_exodus = torch.sqrt(((out_exodus)**2).mean())
        print(f"RMSE: {rmse} (rms exo: {rms_exodus}")
        abs_dev = torch.abs(out_exodus-out_slayer)
        max_dev = torch.max(abs_dev)
        print(f"Max deviation: {max_dev}")
        median = torch.quantile(abs_dev, q=0.5)
        q90 = torch.quantile(abs_dev, q=0.9)
        print(f"Median: {median}, .9 quantile: {q90}")
        if i == 2:
            break

        
