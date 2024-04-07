#!/bin/sh
DATA_DIR="../data/mnist"

rm -r exp/* ## clear out experimental directory
python train_model.py  --dataX="$DATA_DIR/trainX.npy" --dataY="$DATA_DIR/trainY.npy" --devX="$DATA_DIR/testX.npy" --devY="$DATA_DIR/testY.npy" 
