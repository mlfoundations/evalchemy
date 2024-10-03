source ~/.bash_profile

model_name=gpt-3.5-turbo

method_name="SCComplexCoT"

ds_name="GSM8K"
#ds_name="MATH"

python main_forward_reasoning.py --part part1 --eng $model_name --ds $ds_name --method $method_name --temp 0.7 --num_repeat 2 --batch_size 500 --time_out 30 --num_proc 20