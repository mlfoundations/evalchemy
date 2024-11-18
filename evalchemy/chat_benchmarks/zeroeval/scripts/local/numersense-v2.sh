bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8
bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8 -r "sampling" -t 1.0

bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8 
bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8  -r "sampling" -t 1.0

bash zero_eval_local.sh -d numersense-v2 -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4
bash zero_eval_local.sh -d numersense-v2 -m meta-llama/Meta-Llama-3-70B-Instruct -p Meta-Llama-3-70B-Instruct -s 1 


bash zero_eval_api.sh -d numersense-v2 -f openai -m deepseek-chat -p deepseek-v2-chat-0628 -s 8

bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-3.5-turbo-0125 -p gpt-3.5-turbo-0125 -s 8

bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-4-0314 -p gpt-4-0314 -s 8

bash zero_eval_api.sh -d numersense-v2 -f google -m google/gemini-1.5-pro -p gemini-1.5-pro -s 8
bash zero_eval_api.sh -d numersense-v2 -f google -m google/gemini-1.5-flash -p gemini-1.5-flash -s 8
bash zero_eval_api.sh -d numersense-v2 -f anthropic -m anthropic/claude-3-5-sonnet-20240620 -p claude-3-5-sonnet-20240620 -s 8 

# python src/evaluation/gsm_eval.py numersense-v2


bash zero_eval_api.sh -d numersense-v2 -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8 -r "greedy@no_cot"