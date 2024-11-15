MODEL=$1 # your model name 
bash evaluation/run_eval_v2_batch.sh $MODEL claude-3-haiku-20240307 # pairwise eval with Claude-3-Opus 
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307/$MODEL.batch-submit.jsonl 

# python src/openai_batch_eval/check_batch_status_with_model_name.py $MODEL