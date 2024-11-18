# bash zero_eval_local.sh -d gsm -m google/gemma-2-2b-it -p gemma-2-2b-it -s 4 -f hf
# wait 
# bash zero_eval_local.sh -d mmlu-redux -m google/gemma-2-2b-it -p gemma-2-2b-it -s 4 -f hf
# wait 
# bash zero_eval_local.sh -d zebra-grid -m google/gemma-2-2b-it -p gemma-2-2b-it -s 4 -f hf
# wait 


bash zero_eval_local.sh -d crux -m google/gemma-2-2b-it -p gemma-2-2b-it -s 4 -f hf
wait 