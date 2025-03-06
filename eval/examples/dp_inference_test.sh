#!/bin/bash

DATASET="mlfoundations-dev/REASONING_evalchemy"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
GLOBAL_SIZE=128
START_RANK=0

# Run all commands in parallel
CUDA_VISIBLE_DEVICES=0 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $START_RANK --repo_id $DATASET --model_name $MODEL_NAME > log_rank$START_RANK.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 1)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 1)).txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 2)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 2)).txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 3)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 3)).txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 4)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 4)).txt 2>&1 &

CUDA_VISIBLE_DEVICES=5 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 5)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 5)).txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 6)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 6)).txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 python eval/examples/external_inference_sharded_vllm.py --global_size $GLOBAL_SIZE --rank $((START_RANK + 7)) --repo_id $DATASET --model_name $MODEL_NAME > log_rank$((START_RANK + 7)).txt 2>&1 &

# Wait for all background processes to complete
wait

echo "All processes completed!"