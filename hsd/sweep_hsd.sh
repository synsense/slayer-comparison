#!/usr/bin/bash

# for tau in 30 100000;
# for rand_seed in 24 42;
for scale in 0.01 0.1;
do
    python train_hsd.py --max_epochs=200 --track_grad_norm=2 --tau_mem=100000 --scale_grad=$scale
done
