#!/bin/bash
GPU_ID=1 #0

N_SAMPLES=10000
DISABLE_ADAPT_AT_EVAL=False

EXP_DIR="exp_trstdp/"
MODEL="trstdp"
#EXP_DIR="exp_evstdp/"
#MODEL="evstdp"
DEV_X="../../data/mnist/trainX.npy" # validX.npy
DEV_Y="../../data/mnist/trainY.npy" # validY.npy
PARAM_SUBDIR="/custom_snapshot2"
#PARAM_SUBDIR="/custom"

## eval model
CUDA_VISIBLE_DEVICES=$GPU_ID python bind_labels.py --dataX=$DEV_X  --dataY=$DEV_Y  \
                                                   --model_type=$MODEL \
                                                   --model_dir=$EXP_DIR$MODEL \
                                                   --n_samples=$N_SAMPLES \
                                                   --exp_dir=$EXP_DIR \
                                                   --disable_adaptation=$DISABLE_ADAPT_AT_EVAL \
                                                   --param_subdir=$PARAM_SUBDIR
