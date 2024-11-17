```bash
bash zero_eval_local.sh -d alpaca_eval -m Qwen/Qwen2-0.5B-Instruct -p Qwen2-0.5B-Instruct -s 4 -r sampling -t 0.5 -o 1.0 -e 1.1
bash zero_eval_local.sh -d alpaca_eval -m yuchenlin/Rex-v0.1-0.5B -p Rex-v0.1-0.5B -s 4 -r sampling -t 0.5 -o 1.0 -e 1.1 
bash zero_eval_local.sh -d alpaca_eval -m Qwen/Qwen2-1.5B-Instruct -p Qwen2-1.5B-Instruct -s 4 -r sampling -t 0.5 -o 1.0 -e 1.1
bash zero_eval_local.sh -d alpaca_eval -m yuchenlin/Rex-v0.1-1.5B -p Rex-v0.1-1.5B -s 4 -r sampling -t 0.5 -o 1.0 -e 1.1 


bash zero_eval_local.sh -d alpaca_eval -m Qwen/Qwen2-0.5B-Instruct -p Qwen2-0.5B-Instruct -s 4 -r greedy -t 0 -e 1.1
bash zero_eval_local.sh -d alpaca_eval -m yuchenlin/Rex-v0.1-0.5B -p Rex-v0.1-0.5B -s 4 -r greedy -t 0 -e 1.1 
bash zero_eval_local.sh -d alpaca_eval -m Qwen/Qwen2-1.5B-Instruct -p Qwen2-1.5B-Instruct -s 4 -r greedy -t 0 -e 1.1
bash zero_eval_local.sh -d alpaca_eval -m yuchenlin/Rex-v0.1-1.5B -p Rex-v0.1-1.5B -s 4 -r greedy -t 0 -e 1.1 
```


```bash
# model_name="Rex-v0.1-0.5B"
model_name="Qwen2-0.5B-Instruct"
# model_name="Rex-v0.1-1.5B"
# model_name="Qwen2-1.5B-Instruct"
python evaluation/alpaca_eval_formatting.py result_dirs/alpaca_eval/sampling/${model_name}.json
alpaca_eval --model_outputs result_dirs/alpaca_eval/sampling/${model_name}.ae.json --reference_outputs result_dirs/alpaca_eval/gpt-3.5-turbo-0301.ae.json
```