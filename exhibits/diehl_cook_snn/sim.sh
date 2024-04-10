#!/bin/sh

################################################################################
# Simulate the PCN on the MNIST database
################################################################################

#DATA_DIR="../data/baby_mnist"
DATA_DIR="../data/mnist"

rm -r exp/* ## clear out experimental directory
python train_dcsnn.py --dataX="$DATA_DIR/trainX.npy" --n_samples=10000 \
                      --n_iter=1 --verbosity=0
