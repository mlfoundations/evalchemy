# Starting 0612, we will use "gpt-4o-2024-05-13" for evaluation
MODEL=$1 # your model name
bash evaluation/run_eval_v2_batch.score.sh $MODEL  
python src/openai_batch_eval/submit_batch.py eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/$MODEL.batch-submit.jsonl

# python src/openai_batch_eval/check_batch_status_with_model_name.py $MODEL