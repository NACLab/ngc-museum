#!/bin/sh
################################################################################
# Simulate the DC-SNN on the MNIST database
################################################################################
DATA_DIR="../../data/mnist"

rm -r exp/* ## clear out experimental directory
python train_patch_snn.py --dataX="$DATA_DIR/trainX.npy" --n_samples=1000 \
                          --n_iter=1 --verbosity=0
