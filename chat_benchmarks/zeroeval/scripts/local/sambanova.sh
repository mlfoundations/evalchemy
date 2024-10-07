# bash zero_eval_api.sh -f openai -d gsm -m meta-llama/Meta-Llama-3.1-405B-Instruct@hyperbolic -p Llama-3.1-405B-Instruct-BF16 -s 1
# wait
# bash zero_eval_api.sh -f openai -d crux -m meta-llama/Meta-Llama-3.1-405B-Instruct@hyperbolic -p Llama-3.1-405B-Instruct-BF16 -s 1
# wait
# bash zero_eval_api.sh -f openai -d mmlu-redux -m meta-llama/Meta-Llama-3.1-405B-Instruct@hyperbolic -p Llama-3.1-405B-Instruct-BF16 -s 1
# wait
# bash zero_eval_api.sh -f openai -d zebra-grid -m meta-llama/Meta-Llama-3.1-405B-Instruct@hyperbolic -p Llama-3.1-405B-Instruct-BF16 -s 1
# wait

# bash zero_eval_api.sh -f openai -d math-l5 -m Meta-Llama-3.1-405B-Instruct@sambanova -p Llama-3.1-405B-Inst@sambanova -s 1 -x 2048
# wait
bash zero_eval_api.sh -f openai -d crux -m Meta-Llama-3.1-405B-Instruct@sambanova -p Llama-3.1-405B-Inst@sambanova -s 1 -x 2048
wait
bash zero_eval_api.sh -f openai -d mmlu-redux -m Meta-Llama-3.1-405B-Instruct@sambanova -p Llama-3.1-405B-Inst@sambanova -s 1 -x 2048
wait
bash zero_eval_api.sh -f openai -d zebra-grid -m Meta-Llama-3.1-405B-Instruct@sambanova -p Llama-3.1-405B-Inst@sambanova -s 1 -x 2048
wait
bash zero_eval_api.sh -f openai -d gsm -m Meta-Llama-3.1-405B-Instruct@sambanova -p Llama-3.1-405B-Inst@sambanova -s 1 -x 2048
wait



# bash zero_eval_api.sh -f openai -d math-l5 -m Meta-Llama-3.1-70B-Instruct@hyperbolic -p Llama-3.1-70B-Instruct-BF16 -s 1
