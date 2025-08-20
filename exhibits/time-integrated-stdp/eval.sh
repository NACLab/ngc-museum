#!/bin/bash
GPU_ID=1 #0

EXP_DIR="exp_trstdp/"
MODEL="trstdp"
#EXP_DIR="exp_evstdp/"
#MODEL="evstdp"
PARAM_SUBDIR="/custom_snapshot2"
#PARAM_SUBDIR="/custom"
DISABLE_ADAPT_AT_EVAL=False ## set to true to turn off eval-time adaptive thresholds
MAKE_CLUSTER_PLOT=False

DEV_X="../../data/mnist/testX.npy" # validX.npy
DEV_Y="../../data/mnist/testY.npy" # validY.npy

# eval model
CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py --dataX=$DEV_X  --dataY=$DEV_Y  \
                                            --model_type=$MODEL --model_dir=$EXP_DIR$MODEL \
                                            --label_fname=$EXP_DIR"binded_labels.npy" \
                                            --exp_dir=$EXP_DIR \
                                            --disable_adaptation=$DISABLE_ADAPT_AT_EVAL \
                                            --param_subdir=$PARAM_SUBDIR \
                                            --make_cluster_plot=$MAKE_CLUSTER_PLOT
