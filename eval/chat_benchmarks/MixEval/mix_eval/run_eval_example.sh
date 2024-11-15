export MODEL_PARSER_API=<YOUR_OAI_API>
export k_oai=<YOUR_OAI_API
export k_snova=<YOUR_SNOVA_API>

# This runs evaluation on the MixEval benchmark using the llama_3_8b model with the gpt-3.5-turbo-0125 as judge (default on MixEval).
# Max GPU memory and batch size should be changed depending on the GPU memory available.
# Api parallel num should be changed depending on the API limits.
python mix_eval/evaluate.py \
    --model_name llama_3_8b \
    --benchmark mixeval \
    --version 2024-06-01 \
    --batch_size 16 \
    --max_gpu_memory 40GiB \
    --output_dir mix_eval/data/model_responses/ \
    --api_parallel_num 16 \
    --multichoice_judge gpt-3.5-turbo-0125 \
    --freeform_judge gpt-3.5-turbo-0125 \

# The script will look for MODEL_PARSER_API by default and use that if it is set, unset it to use the SNOVA_API instead.
unset MODEL_PARSER_API
export SNOVA_API=<YOUR_SNOVA_API>

# This runs evaluation on the MixEval benchmark using the llama_3_8b model with the Meta-Llama-3.1-405B-Instruct as judge.
python mix_eval/evaluate.py \
    --model_name llama_3_8b \
    --benchmark mixeval \
    --version 2024-06-01 \
    --batch_size 1 \
    --max_gpu_memory 40GiB \
    --output_dir mix_eval/data/model_responses/ \
    --api_parallel_num 1 \
    --multichoice_judge Meta-Llama-3.1-405B-Instruct \
    --freeform_judge Meta-Llama-3.1-405B-Instruct \
