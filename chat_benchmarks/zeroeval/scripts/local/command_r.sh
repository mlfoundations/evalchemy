# bash zero_eval_api.sh -f cohere -d gsm -m cohere/command-r -p command-r -s 8
# wait 
# bash zero_eval_api.sh -f cohere -d mmlu-redux -m cohere/command-r -p command-r -s 8
# wait 
# bash zero_eval_api.sh -f cohere -d zebra-grid -m cohere/command-r -p command-r -s 8
# wait 

bash zero_eval_api.sh -f cohere -d gsm -m cohere/command-r-plus -p command-r-plus -s 8
wait 
bash zero_eval_api.sh -f cohere -d mmlu-redux -m cohere/command-r-plus -p command-r-plus -s 8
wait 
bash zero_eval_api.sh -f cohere -d zebra-grid -m cohere/command-r-plus -p command-r-plus -s 8
wait 


