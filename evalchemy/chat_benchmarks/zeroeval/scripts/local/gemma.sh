export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASHINFER
bash zero_eval_local.sh -d mmlu-redux -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1
bash zero_eval_local.sh -d gsm -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1
bash zero_eval_local.sh -d zebra-grid -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1