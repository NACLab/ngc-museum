#!/bin/sh
################################################################################
# Simulate a Kohonen map on the MNIST database
################################################################################
DATA_DIR="../../data/mnist"

rm -r exp_out/* ## clear out experimental directory
python fit_som.py --dataX="$DATA_DIR/trainX.npy" --n_epochs=5 --verbosity=0

