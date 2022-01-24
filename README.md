# slayer-comparison

## install dependencies
1. Install Sinabs from the dev/0.3/dev branch
2. Install Sinabs-slayer from the dev/slayer_experiments branch
3. Install the original SLAYER: https://github.com/bamsumit/slayerPytorch
3. Install additional requirements: `pip install -r requirements.txt`

## Experiments
We use PyTorch lightning to set up flexible experiments. Enter one of the example lines below on the command line to start an experiment and a tensorboard instance will automatically be started, which logs to './lightning_logs'. 

### NMNIST
`python train_nmnist.py --batch_size=128 --first_saccade_only --gpus=1 --n_time_bins=100 --dataset_fraction=0.5`

### Poisson train
`python train_poisson.py --gpus=1 --max_epochs=2000`

### For Lava experiments
3. Install Lava. Instructions [here](https://github.com/lava-nc/lava)
4. Install Lava-dl, instructions [here](https://github.com/lava-nc/lava-dl)