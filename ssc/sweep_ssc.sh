#!/usr/bin/bash

for rand_seed in 123 1234 12345;
do
    python train_nmnist.py --batch_size=128 --track_grad_norm=2 --tau_mem=100000. --max_epochs=200 --rand_seed=$rand_seed
done

for scale in 0.01 0.1;
do
    python train_nmnist.py --batch_size=128 --track_grad_norm=2 --tau_mem=100000. --max_epochs=200 --scale_grad=$scale
done