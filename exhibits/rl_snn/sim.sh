#!/bin/bash

gpu_id=0 ## set index to point to available GPU on system

seed_array=(1234 ) #44 59 816) ## experimental noise seeds
trial_id=0 ## experimental trial identifier
results_dir=results/ratmaze/ ## results output directory to store arrays

## run experiments across seed array above
for i in "${seed_array[@]}"
do
   echo ">> Running trial $i for SNN agent <<"
   CUDA_VISIBLE_DEVICES=$gpu_id python sim_ratmaze.py --seed=$i --is_random=False \
	                                                    --results_dir=$results_dir --is_verbose=True
	 echo ">> Running trial $i for Random agent <<"
   CUDA_VISIBLE_DEVICES=$gpu_id python sim_ratmaze.py --seed=$i --is_random=True \
	                                                    --results_dir=$results_dir --is_verbose=True
   trial_id=$((trial_id+1)) # update trial identifier for next experiment
done


echo ">> Plotting results to disk..."
seed_string="${seed_array[*]}"
python plot_results.py --result_type="returns" --results_dir=$results_dir --seeds="$seed_string"
python plot_results.py --result_type="completes" --results_dir=$results_dir --seeds="$seed_string"
