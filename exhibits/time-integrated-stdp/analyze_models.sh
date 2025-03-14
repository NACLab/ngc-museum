#!/bin/bash

## get in user-provided program args
GPU_ID=$1 #1
MODEL=$2 # evstdp trstdp tistdp

if [[ "$MODEL" != "evstdp" && "$MODEL" != "trstdp" && "$MODEL" != "tistdp" ]]; then
  echo "Invalid Arg: $MODEL -- only 'evstdp', 'trstdp', 'tistdp' models supported!"
  exit 1
fi
echo " >>>> Setting up $MODEL on GPU $GPU_ID"

SEEDS=(1234 77 811)

PARAM_SUBDIR="/custom"
DISABLE_ADAPT_AT_EVAL=False ## set to true to turn off eval-time adaptive thresholds
MAKE_CLUSTER_PLOT=False #True
REBIND_LABELS=0 ## rebind labels to train model?

N_SAMPLES=50000
DATA_X="../../data/mnist/trainX.npy"
DATA_Y="../../data/mnist/trainY.npy"
DEV_X="../../data/mnist/testX.npy" # validX.npy
DEV_Y="../../data/mnist/testY.npy" # validY.npy
EXTRACT_TRAINING_SPIKES=0 # set to 1 if you want to extract training set codes

for seed in "${SEEDS[@]}"
do
  EXP_DIR="exp_$MODEL""_$seed/"
  echo " > Running Simulation/Model: $EXP_DIR"

  CODEBOOK=$EXP_DIR"training_codes.npy"
  TEST_CODEBOOK=$EXP_DIR"test_codes.npy"
  PLOT_FNAME=$EXP_DIR"codes.jpg"

  if [[ $REBIND_LABELS == 1 ]]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python bind_labels.py --dataX=$DATA_X  --dataY=$DATA_Y  \
                                                       --model_type=$MODEL \
                                                       --model_dir=$EXP_DIR$MODEL \
                                                       --n_samples=$N_SAMPLES \
                                                       --exp_dir=$EXP_DIR \
                                                       --disable_adaptation=$DISABLE_ADAPT_AT_EVAL \
                                                       --param_subdir=$PARAM_SUBDIR
  fi

  ## eval model
#  CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py --dataX=$DEV_X  --dataY=$DEV_Y  \
#                                              --model_type=$MODEL --model_dir=$EXP_DIR$MODEL \
#                                              --label_fname=$EXP_DIR"binded_labels.npy" \
#                                              --exp_dir=$EXP_DIR \
#                                              --disable_adaptation=$DISABLE_ADAPT_AT_EVAL \
#                                              --param_subdir=$PARAM_SUBDIR \
#                                              --make_cluster_plot=$MAKE_CLUSTER_PLOT
  ## call codebook extraction processes
  if [[ $EXTRACT_TRAINING_SPIKES == 1 ]]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_codes.py --dataX=$DATA_X \
                                                         --n_samples=$N_SAMPLES \
                                                         --codebook_fname=$CODEBOOK \
                                                         --model_type=$MODEL \
                                                         --model_fname=$EXP_DIR$MODEL \
                                                         --disable_adaptation=False \
                                                         --param_subdir=$PARAM_SUBDIR
  fi
  CUDA_VISIBLE_DEVICES=$GPU_ID python extract_codes.py --dataX=$DEV_X \
                                                       --codebook_fname=$TEST_CODEBOOK \
                                                       --model_type=$MODEL \
                                                       --model_fname=$EXP_DIR$MODEL \
                                                       --disable_adaptation=False \
                                                       --param_subdir=$PARAM_SUBDIR
  ## visualize latent codes
  CUDA_VISIBLE_DEVICES=$GPU_ID python viz_codes.py --plot_fname=$PLOT_FNAME \
                                                   --codes_fname=$TEST_CODEBOOK \
                                                   --labels_fname=$DEV_Y
done
