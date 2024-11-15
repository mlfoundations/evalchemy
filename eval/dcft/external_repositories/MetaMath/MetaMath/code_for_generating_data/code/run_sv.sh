source ~/.bash_profile

model_name=gpt-3.5-turbo

ds_name="GSM8K"
#ds_name="MATH"

python main_self_verification.py --eng $model_name --ds $ds_name --temp 0.7 --num_repeat 2 --batch_size 500 --time_out 30 --num_proc 20


python main_backward_reasoning.py --part part1 --method_name SV --eng $model_name --ds "${ds_name}_SV" --temp 0.7 --num_repeat 2 --batch_size 500 --time_out 30  --num_proc 20

