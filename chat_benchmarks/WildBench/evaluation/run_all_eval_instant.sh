MODEL=$1 # your model name
bash evaluation/run_eval_v2_instant.score.sh $MODEL # individual scoring 
wait 
bash evaluation/run_eval_v2_instant.sh $MODEL gpt-4-turbo-2024-04-09 # pairwise eval with gpt-4-turbo
wait 
bash evaluation/run_eval_v2_instant.sh $MODEL claude-3-haiku-20240307 # pairwise eval with Claude-3-Opus
wait 
bash evaluation/run_eval_v2_instant.sh $MODEL Llama-2-70b-chat-hf # pairwise eval with Llama-2-70b-chat
wait 
 