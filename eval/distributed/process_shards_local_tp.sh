#!/bin/bash

# SHARDED INFERENCE ARGUMENTS
export GLOBAL_SIZE=8
export MODEL_NAME="open-thoughts/OpenThinker-7B"
export INPUT_DATASET="mlfoundations-dev/evalset_0981"
export OUTPUT_DATASET="results/OpenThinker-7B_evalset_2870"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Print the current configuration
echo "Starting local processing with the following configuration:"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "INPUT_DATASET: ${INPUT_DATASET}"
echo "OUTPUT_DATASET: ${OUTPUT_DATASET}"
echo "GLOBAL_SIZE: ${GLOBAL_SIZE}"

# Ensure output directory exists
mkdir -p ${OUTPUT_DATASET}

# Run each shard on a separate GPU
for i in 0 4; do
  DEVICES=$(seq -s, $i $((i+3)))
  RANK=$((i / 4))
  echo "Launching RANK $RANK on GPUs $DEVICES"
  CUDA_VISIBLE_DEVICES=$DEVICES python eval/distributed/process_shard.py \
    --global_size 2 \
    --rank $RANK \
    --tp 4 \
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