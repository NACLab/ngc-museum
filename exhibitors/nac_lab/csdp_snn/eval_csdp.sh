#!/bin/bash
################################################################################
# Analyze CSDP on target pattern database
################################################################################
GPU_ID=$1

## analyze a saved model on test-set of MNIST database & produce a tSNE plot 
## of its saved rate codes to disk (in folder exp_supervised_mnist/tsne)
DATAX="../../../data/mnist/testX.npy"
DATAY="../../../data/mnist/testY.npy"
MODEL_DIR="exp_supervised_mnist/"
PARAM_SUBDIR="best_params1234"
viz_tsne=True
codebook_name="test_codes" ## file-name of saved rate codeboook created by SNN

CUDA_VISIBLE_DEVICES=$GPU_ID python analyze_csdp.py --dataX=$DATAX \
                                                    --dataY=$DATAY \
                                                    --verbosity=0 \
                                                    --modelDir=$MODEL_DIR \
                                                    --paramDir=$PARAM_SUBDIR \
                                                    --viz_tsne=$viz_tsne \
                                                    --codebookName=$codebook_name
