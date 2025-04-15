#!/bin/bash

# SHARDED INFERENCE ARGUMENTS
export GLOBAL_SIZE=8
export MODEL_NAME="open-thoughts/OpenThinker-7B"
export INPUT_DATASET="mlfoundations-dev/evalset_0981"
export OUTPUT_DATASET="results/OpenThinker-7B_evalset_2870"

# Print the current configuration
echo "Starting local processing with the following configuration:"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "INPUT_DATASET: ${INPUT_DATASET}"
echo "OUTPUT_DATASET: ${OUTPUT_DATASET}"
echo "GLOBAL_SIZE: ${GLOBAL_SIZE}"

# Ensure output directory exists
mkdir -p ${OUTPUT_DATASET}

# Run each shard on a separate GPU
for RANK in {0..7}; do
  # Print config for this rank
  echo -e "\nProcessing RANK: ${RANK}"
  echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DATASET: ${OUTPUT_DATASET}"
  
  # Launch process on specific GPU
  CUDA_VISIBLE_DEVICES=$RANK python eval/distributed/process_shard.py \
    --global_size ${GLOBAL_SIZE} \
    --rank ${RANK} \
    --input_dataset ${INPUT_DATASET} \
    --model_name ${MODEL_NAME} \
    --output_dataset ${OUTPUT_DATASET} &
    
  # Add a small delay between launches to avoid race conditions
  sleep 2
done

echo "All processes launched. Waiting for completion..."

# Wait for all background processes to complete
wait

echo "All processes have completed"