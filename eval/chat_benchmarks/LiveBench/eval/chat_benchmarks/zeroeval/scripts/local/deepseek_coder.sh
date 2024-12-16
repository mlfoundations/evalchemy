# bash zero_eval_api.sh -f openai -d gsm -m deepseek-coder -p DeepSeek-Coder-V2-0724 -s 8
# bash zero_eval_api.sh -f openai -d mmlu-redux -m deepseek-coder -p DeepSeek-Coder-V2-0724 -s 8
# bash zero_eval_api.sh -f openai -d zebra-grid -m deepseek-coder -p DeepSeek-Coder-V2-0724 -s 8




bash zero_eval_api.sh -f openai -d mmlu-redux -m deepseek-chat -p deepseek-v2.5-0908 -s 8
wait
bash zero_eval_api.sh -f openai -d zebra-grid -m deepseek-chat -p deepseek-v2.5-0908 -s 8
wait
bash zero_eval_api.sh -f openai -d crux -m deepseek-chat -p deepseek-v2.5-0908 -s 8
wait
# bash zero_eval_api.sh -f openai -d math-l5 -m deepseek-chat -p deepseek-v2.5-0908 -s 8 
# wait 
bash zero_eval_api.sh -f openai -d gsm -m deepseek-chat -p deepseek-v2.5-0908 -s 8
wait