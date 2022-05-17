#!/bin/bash

for rand_seed in 123 1234 12345;
do
    python train_nmnist.py --batch_size=64 --gpus=1 --architecture='small' --track_grad_norm=2 --tau_mem=100000. --max_epochs=75 --rand_seed=$rand_seed
done

for scale in 0.01 0.1;
do
    python train_nmnist.py --batch_size=64 --gpus=1 --architecture='small' --track_grad_norm=2 --tau_mem=100000. --max_epochs=75 --scale_grad=$scale
done