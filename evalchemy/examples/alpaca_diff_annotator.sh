accelerate launch --num-processes 1 --num-machines 1 -m evalchemy.eval \
    --model hf \
    --task alpaca_eval \
    --model_args 'pretrained=mistralai/Mistral-7B-Instruct-v0.3' \
    --batch_size 16 \
    --output_path logs \
    --annotator_model gpt-4o-mini-2024-07-18
    