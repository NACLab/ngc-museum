#!/bin/bash

################################################################################
# Simulate CSDP on target pattern database
################################################################################
GPU_ID=$1 ## GPU identifier (0 should be used if only one GPU exists)

## demo model sizing/setup/data parameters
SEED=1234
DATASET="mnist"
NUM_ITER=10 ## number of epochs
DATA_DIR="../../../data/"$DATASET ## configure data directory pointer
DEV_NAME="valid" ## point to dev-set
ALGO_TYPE="supervised"
NZ1=3000 ## number LIFs in layer 1
NZ2=600 ## number LIFs in layer 2

## create base folders (if need be) and run the simulation/experiment
mkdir -p logging
rm -f logging/* ## clear out any logs in logging directory
EXP_DIR="exp_"$ALGO_TYPE"_"$DATASET
rm -rf "$EXP_DIR/"* ## clear out experimental directory
CUDA_VISIBLE_DEVICES=$GPU_ID python train_csdp.py --dataX="$DATA_DIR/trainX.npy" \
                                                  --dataY="$DATA_DIR/trainY.npy" \
                                                  --devX=$DATA_DIR/$DEV_NAME"X.npy" \
                                                  --devY=$DATA_DIR/$DEV_NAME"Y.npy" \
                                                  --algo_type=$ALGO_TYPE \
                                                  --num_iter=$NUM_ITER \
                                                  --verbosity=0 --seed=$SEED \
                                                  --exp_dir=$EXP_DIR --nZ1=$NZ1 --nZ2=$NZ2
