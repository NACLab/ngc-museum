#!/bin/bash

## get in user-provided program args
GPU_ID=$1 #1
MODEL=$2 # evstdp trstdp tistdp stdp

if [[ "$MODEL" != "evstdp" && "$MODEL" != "tistdp" ]]; then
  echo "Invalid Arg: $MODEL -- only 'tistdp', 'evstdp' models supported!"
  exit 1
fi
echo " >>>> Setting up $MODEL on GPU $GPU_ID"

PARAM_SUBDIR="/custom"
SEED=1234 #(1234 77 811)
N_SAMPLES=5000 #1000 #50000
DATA_X="../../data/mnist/trainX.npy"
DATA_Y="../../data/mnist/trainY.npy"
#DEV_X="../../data/mnist/testX.npy" # validX.npy
#DEV_Y="../../data/mnist/testY.npy" # validY.npy

EXP_DIR="exp_$MODEL""_$SEED/"
echo " > Running Simulation/Model: $EXP_DIR"

## train model
CUDA_VISIBLE_DEVICES=$GPU_ID python assemble_patterns.py --dataX=$DATA_X  --dataY=$DATA_Y \
                                                         --n_samples=$N_SAMPLES --exp_dir=$EXP_DIR \
                                                         --model_dir=$EXP_DIR$MODEL \
                                                         --param_subdir=$PARAM_SUBDIR \
                                                         --model_type=$MODEL --seed=$SEED
