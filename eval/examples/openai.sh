python -m eval.eval \
    --model openai-completions \                
    --tasks alpaca_eval \    
    --model_args "model=gpt-4o-mini-2024-07-18,num_concurrent=32" \  
    --batch_size 16 \         
    --output_path logs         
