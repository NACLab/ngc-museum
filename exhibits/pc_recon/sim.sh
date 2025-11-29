#!/bin/sh
################################################################################
# Simulate sparse coding on the MNIST database
################################################################################
DATA_DIR="../../data/mnist"

rm -r exp/* ## clear out experimental directory
python train_pc_recon.py --path_data="$DATA_DIR" --n_iter=10 --n_samples=-1
