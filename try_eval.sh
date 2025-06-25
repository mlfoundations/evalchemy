# python -m eval.eval \
#     --model curator \
#     --tasks GPQADiamond \
#     --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=50000" \
#     --num_fewshot 0 \
#     --batch_size 4 \
#     --limit 1 \
#     --apply_chat_template False \
#     --max_tokens 8000 \
#     --output_path logs/GPQADiamond_results_test          

python -m eval.eval \
    --model curator \
    --tasks AMC23 \
    --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=50000" \
    --limit 3 \
    --max_tokens 200 \
    --output_path logs/AMC23_results_test