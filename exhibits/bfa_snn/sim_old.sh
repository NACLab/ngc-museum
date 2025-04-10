#!/bin/sh
################################################################################
# Simulate the BFA-SNN on the MNIST database
################################################################################
DATA_DIR="../../data/mnist"

rm -r exp/* ## clear out experimental directory
python train_bfasnn_old.py  --dataX="$DATA_DIR/trainX.npy" \
                        --dataY="$DATA_DIR/trainY.npy" \
                        --devX="$DATA_DIR/validX.npy" \
                        --devY="$DATA_DIR/validY.npy" \
                        --verbosity=1
