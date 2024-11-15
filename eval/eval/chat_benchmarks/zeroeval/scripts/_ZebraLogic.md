# Testing LLMs on MMLU_redux with ZeroEval

```bash 
# export VLLM_WORKER_MULTIPROC_METHOD="FLASH_ATTN"
# VLLM_WORKER_MULTIPROC_METHOD=spawn 
bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2-72B-Instruct -p Qwen2-72B-Instruct -s 1
bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2-72B-Instruct -p Qwen2-72B-Instruct -s 1 -r "sampling" -t 0.5
bash zero_eval_local.sh -d zebra-grid -m meta-llama/Meta-Llama-3-70B-Instruct -p Meta-Llama-3-70B-Instruct -s 1 
bash zero_eval_local.sh -d zebra-grid -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4
bash zero_eval_local.sh -d zebra-grid -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4 -r "sampling" -t 0.5
bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2-7B-Instruct -p Qwen2-7B-Instruct -s 4 
bash zero_eval_local.sh -d zebra-grid -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1

bash zero_eval_local.sh -d zebra-grid -m mistralai/mathstral-7B-v0.1 -p mathstral-7B-v0.1 -s 4

bash zero_eval_local.sh -f hf -d zebra-grid -m mistralai/Mistral-Nemo-Instruct-2407 -p Mistral-Nemo-Instruct-2407 -s 4
```

<!-- 
# pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.3/
# VLLM_WORKER_MULTIPROC_METHOD=spawn 
# export VLLM_ATTENTION_BACKEND=FLASHINFER; bash zero_eval_local.sh -d zebra-grid -m google/gemma-2-9b-it -p gemma-2-9b-it -s 4  -b 1
# export VLLM_ATTENTION_BACKEND=FLASHINFER; bash zero_eval_local.sh -d zebra-grid -m google/gemma-2-27b-it -p gemma-2-27b-it -s 1 -b 1 -r "sampling" -t 0.5 
-->



```bash

bash zero_eval_api.sh -d zebra-grid -f openai -m google/gemma-2-9b-it@nvidia -p gemma-2-9b-it@nvidia -s 8
bash zero_eval_api.sh -d zebra-grid -f openai -m google/gemma-2-27b-it@nvidia -p gemma-2-27b-it@nvidia -s 8

bash zero_eval_api.sh -d zebra-grid -f openai -m openai/gpt-3.5-turbo-0125 -p gpt-3.5-turbo-0125 -s 8
bash zero_eval_api.sh -d zebra-grid -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8 
bash zero_eval_api.sh -d zebra-grid -f openai -m openai/gpt-4-0314 -p gpt-4-0314 -s 8
bash zero_eval_api.sh -d zebra-grid -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8 -r "sampling" -t 0.5
bash zero_eval_api.sh -d zebra-grid -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 
bash zero_eval_api.sh -d zebra-grid -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 -r "sampling" -t 0.5

bash zero_eval_api.sh -d zebra-grid -f anthropic -m anthropic/claude-3-5-sonnet-20240620 -p claude-3-5-sonnet-20240620 -s 8
bash zero_eval_api.sh -d zebra-grid -f anthropic -m anthropic/claude-3-5-sonnet-20240620 -p claude-3-5-sonnet-20240620 -s 8 -r "sampling" -t 0.5
wait
bash zero_eval_api.sh -d zebra-grid -f anthropic -m anthropic/claude-3-haiku-20240307 -p claude-3-haiku-20240307 -s 8
wait
bash zero_eval_api.sh -d zebra-grid -f anthropic -m anthropic/claude-3-opus-20240229 -p claude-3-opus-20240229 -s 8
wait
bash zero_eval_api.sh -d zebra-grid -f anthropic -m anthropic/claude-3-sonnet-20240229 -p claude-3-sonnet-20240229 -s 8
wait

bash zero_eval_api.sh -d zebra-grid -f google -m google/gemini-1.5-pro -p gemini-1.5-pro -s 8
bash zero_eval_api.sh -d zebra-grid -f google -m google/gemini-1.5-flash -p gemini-1.5-flash -s 8

bash zero_eval_api.sh -d zebra-grid -f google -m google/gemini-1.5-pro -p gemini-1.5-pro -s 8 -r "sampling" -t 0.5
bash zero_eval_api.sh -d zebra-grid -f google -m google/gemini-1.5-flash -p gemini-1.5-flash -s 8 -r "sampling" -t 0.5

bash zero_eval_api.sh -d zebra-grid -f reka -m reka/reka-flash-20240226 -p reka-flash-20240226 -s 8
bash zero_eval_api.sh -d zebra-grid -f reka -m reka/reka-core-20240501 -p reka-core-20240501 -s 8

bash zero_eval_api.sh -d zebra-grid -f together -m meta-llama/Llama-3-70b-chat-hf -p Llama-3-70b-chat-hf -s 8
bash zero_eval_api.sh -d zebra-grid -f together -m meta-llama/Llama-3-8B-chat-hf -p Llama-3-8B-chat-hf -s 8

bash zero_eval_api.sh -d zebra-grid -f openai -m deepseek-chat -p deepseek-chat -s 8
bash zero_eval_api.sh -d zebra-grid -f openai -m deepseek-coder -p deepseek-coder -s 8

bash zero_eval_api.sh -d zebra-grid -f openai -m yi-large -p yi-large -s 8
bash zero_eval_api.sh -d zebra-grid -f openai -m yi-large-preview -p yi-large-preview -s 8

bash zero_eval_api.sh -d zebra-grid -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Instruct-Turbo -s 8 -x 2048


bash zero_eval_api.sh -d zebra-grid -f google -m google/gemini-1.5-pro-exp-0827 -p gemini-1.5-pro-exp-0827 -s 1

```

 