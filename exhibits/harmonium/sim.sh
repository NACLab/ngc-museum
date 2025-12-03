#!/bin/sh
################################################################################
# Simulate the DC-SNN on the MNIST database
################################################################################
DATA_DIR="../../data/mnist"

#rm -r exp/* ## clear out experimental directory
python sim_harmonium.py --trainX="$DATA_DIR/trainX.npy" \
	                      --devX="$DATA_DIR/validX.npy" \
			                  --verbosity=0

python sample_harmonium.py --dataX="$DATA_DIR/trainX.npy" \
                           --output_dir="exp/samples/"

