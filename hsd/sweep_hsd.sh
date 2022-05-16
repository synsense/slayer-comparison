#!/usr/bin/bash

for tau in 30 60 100000;
do
    python train_hsd.py --max_epochs=100 --track_grad_norm=2 --tau_mem=$tau
done