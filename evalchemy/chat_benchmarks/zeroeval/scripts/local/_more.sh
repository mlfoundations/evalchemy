# bash zero_eval_api.sh -d math-l5 -f openai -m openai/gpt-4-0314 -p gpt-4-0314 -s 8
# wait 
# bash zero_eval_api.sh -d math-l5 -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 
# wait 
# bash zero_eval_api.sh -d math-l5 -f anthropic -m anthropic/claude-3-opus-20240229 -p claude-3-opus-20240229 -s 8
# wait 

# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 
# wait 
# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-4-0314 -p gpt-4-0314 -s 8
# wait 


# bash zero_eval_api.sh -d crux -f openai -m openai/chatgpt-4o-latest -p chatgpt-4o-latest-24-09-07 -s 8
# wait 
# bash zero_eval_api.sh -d mmlu-redux -f openai -m openai/chatgpt-4o-latest -p chatgpt-4o-latest-24-09-07 -s 8 
# wait 
# bash zero_eval_api.sh -d zebra-grid -f openai -m openai/chatgpt-4o-latest -p chatgpt-4o-latest-24-09-07 -s 8 
# wait 

# bash zero_eval_api.sh -f cohere -d crux -m cohere/command-r-plus -p command-r-plus -s 8
# wait 
# bash zero_eval_api.sh -f cohere -d math-l5 -m cohere/command-r-plus -p command-r-plus -s 8
# wait 

# bash zero_eval_api.sh -d math-l5 -f reka -m reka/reka-core-20240501 -p reka-core-20240501 -s 8
# wait 


# bash zero_eval_local.sh -d crux -m Qwen/Qwen2.5-72B-Instruct -p Qwen2.5-72B-Instruct -s 1
# bash zero_eval_local.sh -d math-l5 -m Qwen/Qwen2.5-72B-Instruct -p Qwen2.5-72B-Instruct -s 1
# bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2.5-72B-Instruct -p Qwen2.5-72B-Instruct -s 1
# bash zero_eval_local.sh -d mmlu-redux -m Qwen/Qwen2.5-72B-Instruct -p Qwen2.5-72B-Instruct -s 1

# bash zero_eval_local.sh -d crux -m Qwen/Qwen2.5-32B-Instruct -p Qwen2.5-32B-Instruct -s 1
# bash zero_eval_local.sh -d math-l5 -m Qwen/Qwen2.5-32B-Instruct -p Qwen2.5-32B-Instruct -s 1
# bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2.5-32B-Instruct -p Qwen2.5-32B-Instruct -s 1
# bash zero_eval_local.sh -d mmlu-redux -m Qwen/Qwen2.5-32B-Instruct -p Qwen2.5-32B-Instruct -s 1


# bash zero_eval_local.sh -d crux -m Qwen/Qwen2.5-7B-Instruct -p Qwen2.5-7B-Instruct -s 4
# bash zero_eval_local.sh -d math-l5 -m Qwen/Qwen2.5-7B-Instruct -p Qwen2.5-7B-Instruct -s 4
# bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2.5-7B-Instruct -p Qwen2.5-7B-Instruct -s 4
# bash zero_eval_local.sh -d mmlu-redux -m Qwen/Qwen2.5-7B-Instruct -p Qwen2.5-7B-Instruct -s 4

bash zero_eval_local.sh -d crux -m Qwen/Qwen2.5-3B-Instruct -p Qwen2.5-3B-Instruct -s 4
bash zero_eval_local.sh -d math-l5 -m Qwen/Qwen2.5-3B-Instruct -p Qwen2.5-3B-Instruct -s 4
bash zero_eval_local.sh -d zebra-grid -m Qwen/Qwen2.5-3B-Instruct -p Qwen2.5-3B-Instruct -s 4
bash zero_eval_local.sh -d mmlu-redux -m Qwen/Qwen2.5-3B-Instruct -p Qwen2.5-3B-Instruct -s 4