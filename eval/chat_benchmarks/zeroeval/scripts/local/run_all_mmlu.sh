# bash zero_eval_api.sh -d mmlu-redux -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8 
wait
# bash zero_eval_api.sh -d mmlu-redux -f openai -m openai/gpt-4-0314 -p gpt-4-0314 -s 8
wait
# bash zero_eval_api.sh -d mmlu-redux -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 
wait

# # bash zero_eval_api.sh -d mmlu-redux -f anthropic -m anthropic/claude-3-5-sonnet-20240620 -p claude-3-5-sonnet-20240620 -s 8
wait
# # bash zero_eval_api.sh -d mmlu-redux -f anthropic -m anthropic/claude-3-haiku-20240307 -p claude-3-haiku-20240307 -s 8
wait
# # bash zero_eval_api.sh -d mmlu-redux -f anthropic -m anthropic/claude-3-opus-20240229 -p claude-3-opus-20240229 -s 8
wait
# # bash zero_eval_api.sh -d mmlu-redux -f anthropic -m anthropic/claude-3-sonnet-20240229 -p claude-3-sonnet-20240229 -s 8
wait



bash zero_eval_api.sh -d mmlu-redux -f google -m google/gemini-1.5-pro -p gemini-1.5-pro -s 8
wait
bash zero_eval_api.sh -d mmlu-redux -f google -m google/gemini-1.5-flash -p gemini-1.5-flash -s 8
wait



bash zero_eval_api.sh -d mmlu-redux -f openai -m deepseek-chat -p deepseek-chat -s 8
wait
bash zero_eval_api.sh -d mmlu-redux -f openai -m deepseek-coder -p deepseek-coder -s 8
wait


bash zero_eval_api.sh -d mmlu-redux -f reka -m reka/reka-flash-20240226 -p reka-flash-20240226 -s 8
wait
bash zero_eval_api.sh -d mmlu-redux -f reka -m reka/reka-core-20240501 -p reka-core-20240501 -s 8
wait
bash zero_eval_api.sh -d mmlu-redux -f openai -m yi-large -p yi-large -s 8
wait
bash zero_eval_api.sh -d mmlu-redux -f openai -m yi-large-preview -p yi-large-preview -s 8
wait


# bash zero_eval_api.sh -d mmlu-redux -f openai -m google/gemma-2-9b-it@nvidia -p gemma-2-9b-it@nvidia -s 8
wait
# bash zero_eval_api.sh -d mmlu-redux -f openai -m google/gemma-2-27b-it@nvidia -p gemma-2-27b-it@nvidia -s 8
wait