
python src/evaluation/mcqa_eval.py mmlu-redux &
python src/evaluation/math_eval.py math-l5 &
python src/evaluation/zebra_grid_eval.py &
python src/evaluation/crux_eval.py &
python src/evaluation/math_eval.py gsm  &
wait 
python src/evaluation/summarize.py

