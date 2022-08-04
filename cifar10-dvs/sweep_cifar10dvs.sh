#!/usr/bin/bash

for rand_seed in 12 123 1234;
do
    python train_cifar10_dvs.py --rand_seed=$rand_seed --batch_size=16 --n_time_bins=8 --gpus=2 --max_epochs=120 --learning_rate=1e-3 --spike_threshold=1. --optimizer=adam --tau_mem=3. --augmentation
done
