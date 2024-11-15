MODEL=$1 # your model name
bash evaluation/run_eval_v2_batch.score.sh $MODEL # individual scoring with GPT-4O (since June 13, 2024)
bash evaluation/run_eval_v2_batch.sh $MODEL gpt-4-turbo-2024-04-09 # pairwise eval with gpt-4-turbo
bash evaluation/run_eval_v2_batch.sh $MODEL claude-3-haiku-20240307 # pairwise eval with Claude-3-Opus
bash evaluation/run_eval_v2_batch.sh $MODEL Llama-2-70b-chat-hf # pairwise eval with Llama-2-70b-chat


python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/$MODEL.batch-submit.jsonl
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307/$MODEL.batch-submit.jsonl
python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf/$MODEL.batch-submit.jsonl
# python src/openai_batch_eval/submit_batch.py eval_results/v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09/$MODEL.batch-submit.jsonl
python src/openai_batch_eval/submit_batch.py eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/$MODEL.batch-submit.jsonl


# python src/openai_batch_eval/check_batch_status_with_model_name.py $MODEL