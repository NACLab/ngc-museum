#!/bin/bash

## get in user-provided program args
GPU_ID=$1 #1
MODEL=$2 # evstdp trstdp tistdp stdp

if [[ "$MODEL" != "evstdp" && "$MODEL" != "trstdp" && "$MODEL" != "tistdp" ]]; then
  echo "Invalid Arg: $MODEL -- only 'evstdp', 'trstdp', 'tistdp' models supported!"
  exit 1
fi
echo " >>>> Setting up $MODEL on GPU $GPU_ID"

SEEDS=(1234) # 77 811)

PARAM_SUBDIR="/custom"
DISABLE_ADAPT_AT_EVAL=False ## set to true to turn off eval-time adaptive thresholds

N_SAMPLES=50000
DATA_X="../../data/mnist/trainX.npy"
DATA_Y="../../data/mnist/trainY.npy"

for seed in "${SEEDS[@]}"
do
  EXP_DIR="final_case1_results/exp_$MODEL""_$seed/"
  echo " > Running Simulation/Model: $EXP_DIR"

  CODEBOOK=$EXP_DIR"training_codes.npy"

  CUDA_VISIBLE_DEVICES=$GPU_ID python extract_codes.py --dataX=$DATA_X \
                                                         --n_samples=$N_SAMPLES \
                                                         --codebook_fname=$CODEBOOK \
                                                         --model_type=$MODEL \
                                                         --model_fname=$EXP_DIR$MODEL \
                                                         --disable_adaptation=$DISABLE_ADAPT_AT_EVAL \
                                                         --param_subdir=$PARAM_SUBDIR
done
