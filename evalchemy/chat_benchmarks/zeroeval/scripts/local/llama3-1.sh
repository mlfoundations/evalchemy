# bash zero_eval_api.sh -d zebra-grid -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Instruct-Turbo -s 8 -x 2048
# bash zero_eval_api.sh -d zebra-grid -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Instruct-Turbo -s 8 -x 2048 -r "sampling" -t 0.5
# bash zero_eval_api.sh -d gsm -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Instruct-Turbo -s 8 -x 2048
# bash zero_eval_api.sh -d mmlu-redux -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Instruct-Turbo -s 8 -x 2048
# bash zero_eval_api.sh -d mmlu-redux -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Instruct-Turbo -s 8 -x 2048 -r "sampling" -t 0.5




# bash zero_eval_local.sh -d gsm -m meta-llama/Meta-Llama-3.1-8B-Instruct -p Meta-Llama-3.1-8B-Instruct -s 4
# wait 
# bash zero_eval_local.sh -d mmlu-redux -m meta-llama/Meta-Llama-3.1-8B-Instruct -p Meta-Llama-3.1-8B-Instruct -s 4
# wait 
# bash zero_eval_local.sh -d zebra-grid -m meta-llama/Meta-Llama-3.1-8B-Instruct -p Meta-Llama-3.1-8B-Instruct -s 4
# wait 


bash zero_eval_local.sh -d gsm -m meta-llama/Meta-Llama-3.1-70B-Instruct -p Meta-Llama-3.1-70B-Instruct -s 1
wait 
bash zero_eval_local.sh -d mmlu-redux -m meta-llama/Meta-Llama-3.1-70B-Instruct -p Meta-Llama-3.1-70B-Instruct -s 1 -b 1
wait 
bash zero_eval_local.sh -d zebra-grid -m meta-llama/Meta-Llama-3.1-70B-Instruct -p Meta-Llama-3.1-70B-Instruct -s 1 -b 1
wait 