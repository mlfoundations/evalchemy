python -m eval.eval \
    --model curator \
    --tasks arabicmmlu_stem \
    --model_args "base_url=http://10.7.60.170/vllm/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=20000" \
    --num_fewshot 3 \
    --batch_size 4 \
    --limit 1 \
    --apply_chat_template False \
    --max_tokens 1000 \
    --output_path logs/qwen3-8b_arabicmmlu_stem_results_test.json
