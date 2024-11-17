num_gpus=8
accelerate launch --num-processes $num_gpus --multi-gpu --num-machines 1 -m eval.eval \
    --model hf \
    --task alpaca_eval \
    --model_args 'pretrained=mistralai/Mistral-7B-Instruct-v0.3' \
    --batch_size 16 \
    --output_path logs 
    
