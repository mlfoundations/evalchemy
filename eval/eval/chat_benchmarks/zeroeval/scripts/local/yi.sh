bash zero_eval_local.sh -d mmlu-redux -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1
wait 
bash zero_eval_local.sh -d zebra-grid -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1
wait 
bash zero_eval_local.sh -d gsm -m 01-ai/Yi-1.5-34B-Chat -p Yi-1.5-34B-Chat -s 1
wait 

bash zero_eval_local.sh -d mmlu-redux -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
wait 
bash zero_eval_local.sh -d zebra-grid -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
wait 
bash zero_eval_local.sh -d gsm -m 01-ai/Yi-1.5-9B-Chat -p Yi-1.5-9B-Chat -s 4
wait 