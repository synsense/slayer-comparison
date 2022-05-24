# slayer-comparison

## install dependencies
1. Install the original [pytorchSLAYER package](https://github.com/bamsumit/slayerPytorch)
2. Install EXODUS from the sinabs-exodus folder
3. Install additional requirements: `pip install -r requirements.txt`

## Experiments
We use PyTorch lightning to set up flexible experiments. Enter one of the example lines below on the command line to start an experiment and a tensorboard instance will automatically be started, which logs to './lightning_logs'. All the training scripts supportcommand line arguments for controlling various aspects of the training procedure. Run the script with `--help` as argument to see a list of all available options.

### DVS Gesture
`python dvs_gesture/train_dvs_gesture.py --gpus=1 --max_epochs=100`

Additional arguments to those listed in `help`, such as `max_epochs` can be provided and will be passed directly to pytorch lightning `Trainer` class. See [pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) for further details.

Alternatively parameter sweeps can be run with the following command:
`python dvs_gesture/sweep_dvs_gesture.py`

### HSD
`python hsd/train_hsd.py
`
### SSC
`python ssc/train_ssc.py`

