# Custom benchmarks
# AMC23
python -m eval.eval \
    --model curator \
    --tasks MATH500 \
    --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=50000" \
    --limit 1 \
    --max_tokens 2048 \
    --output_path logs/MATH500_results_test

# GPQADiamond
python -m eval.eval \
    --model curator \
    --tasks GPQADiamond \
    --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=50000" \
    --limit 1 \
    --apply_chat_template False \
    --max_tokens 2048 \
#    --debug \
    --output_path logs/GPQADiamond_results_test          

# MATH500
python -m eval.eval \
    --model curator \
    --tasks MATH500 \
    --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=50000" \
    --limit 1 \
    --max_tokens 2048 \
    --output_path logs/MATH500_results_test