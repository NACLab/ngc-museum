#!/bin/bash

## get in user-provided program args
GPU_ID=$1 #1
MODEL=$2 # evstdp trstdp tistdp stdp

if [[ "$MODEL" != "evstdp" && "$MODEL" != "tistdp" ]]; then
  echo "Invalid Arg: $MODEL -- only 'tistdp', 'evstdp' models supported!"
  exit 1
fi
echo " >>>> Setting up $MODEL on GPU $GPU_ID"

SEEDS=(1234)
N_ITER=10
N_SAMPLES=50000 #10000 #5000 #1000 #50000
BIND_COUNT=10000
BIND_TARGET=$((($N_ITER - 1) * $N_SAMPLES + ($N_SAMPLES - $BIND_COUNT)))

DATA_X="../../data/mnist/trainX.npy"
DATA_Y="../../data/mnist/trainY.npy"
#DEV_X="../../data/mnist/testX.npy" # validX.npy
#DEV_Y="../../data/mnist/testY.npy" # validY.npy

if (( N_ITER * N_SAMPLES < BIND_COUNT )) ; then
  echo "Not enough samples to reach bind target!"
  exit 1
fi

for seed in "${SEEDS[@]}"
do
  EXP_DIR="exp_$MODEL""_$seed/"
  echo " > Running Simulation/Model: $EXP_DIR"

  rm -r $EXP_DIR*
  ## train model
  CUDA_VISIBLE_DEVICES=$GPU_ID python patched_train.py --dataX=$DATA_X  --dataY=$DATA_Y \
                                                       --n_iter=$N_ITER --bind_target=$BIND_TARGET \
                                                       --n_samples=$N_SAMPLES --exp_dir=$EXP_DIR \
                                                       --model_type=$MODEL --seed=$seed
done
