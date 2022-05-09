from dvs_gesture_model import SlayerNetwork, ExodusNetwork
from data_modules.dvs_gesture import DVSGesture
import torch

kwargs_model = dict(
    batch_size=8,
    tau_mem=10,
    spike_threshold=1,
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
    print(f"Batch {i}", end="\r")
    out_exodus = exodus_model(inp.cuda())
    out_slayer = slayer_model(inp.cuda())
    assert torch.allclose(out_exodus, out_slayer, rtol=1e-6, atol=1e-5)
