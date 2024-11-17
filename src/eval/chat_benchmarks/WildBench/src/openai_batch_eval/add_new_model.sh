MODEL="gpt-4-0125-preview "
bash evaluation/run_eval_v2_batch.score.sh $MODEL gpt-4-turbo-2024-04-09 
bash evaluation/run_eval_v2_batch.sh $MODEL gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
bash evaluation/run_eval_v2_batch.sh $MODEL Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# python src/openai_batch_eval/submit_all.py pairwise
# python src/openai_batch_eval/submit_all.py pairwise-llama
# python src/openai_batch_eval/submit_all.py score
# python src/openai_batch_eval/check_batch_status.py pairwise.v2
# python src/openai_batch_eval/check_batch_status.py score.v2
# python src/openai_batch_eval/check_batch_status.py pairwise.v2-llama