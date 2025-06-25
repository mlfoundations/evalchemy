# Reasoning
python -m eval.eval \
  --model curator \
  --tasks GPQADiamond,AMC23,MATH500,hellaswag_ar \
  --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=20000" \
  --limit 100 \
  --apply_chat_template False \
  --max_tokens 2000 \
  --output_path logs/reasoning_results

# Non-reasoning
python -m eval.eval \
  --model curator \
  --tasks piqa_ar,arabicmmlu,aexams,copa_ar \
  --model_args "base_url=http://10.7.60.89:10001/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=60,max_tokens_per_minute=20000" \
  --limit 100 \
  --apply_chat_template False \
  --max_tokens 500 \
  --output_path logs/non_reasoning_results