model_name=$1 # model to test 
# by default use gpt-3.5-turbo-0125 as ref_name 
# by default use "gpt-4-0125-preview" as gpt_eval_name
# gpt_eval_name=${2:-"gpt-4-turbo-2024-04-09"} # evaluator name  # gpt-4-0125-preview
gpt_eval_name=${2:-"gpt-4o-2024-05-13"} # evaluator name  # gpt-4-0125-preview



# if gpt_eval_name == gpt-4-turbo-2024-04-09; then use `evaluation/eval_template.score.v2.0522.md`
# if gpt_eval_name == gpt-4-0125-preview; then use `evaluation/eval_template.score.v2.md`
if [ $gpt_eval_name == "gpt-4-turbo-2024-04-09" ]; then
    # eval_template="evaluation/eval_template.score.v2.0522.md"
    echo "not supported any more for the concern of cost"
    exit 0
else
    eval_template="evaluation/eval_template.score.v2.md"
fi

eval_folder="eval_results/v2.0625/score.v2/eval=${gpt_eval_name}/"
echo "Evaluating $model_name using $gpt_eval_name with $eval_template"
mkdir -p $eval_folder 
eval_file="${eval_folder}/${model_name}.batch-submit.jsonl"

# judge if the eval_file exists 
if [ -f $eval_file ]; then
    echo "File $eval_file exists, skip generation"
    exit 0
fi


python src/eval.py \
    --batch_mode \
    --action eval \
    --model $gpt_eval_name \
    --mode score \
    --max_words_to_eval -1 \
    --eval_template $eval_template \
    --target_model_name $model_name \
    --eval_output_file $eval_file 

echo "Batch results saved to $eval_file"

## V2.0522

# test_model, ref_model, eval_model

# bash evaluation/run_eval_v2_batch.score.sh claude-3-opus-20240229 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Meta-Llama-3-70B-Instruct gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Meta-Llama-3-8B-Instruct gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh claude-3-sonnet-20240229 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gpt-4o-2024-05-13 gpt-4-turbo-2024-04-09

# bash evaluation/run_eval_v2_batch.score.sh claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gpt-3.5-turbo-0125 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh mistral-large-2402 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09

# bash evaluation/run_eval_v2_batch.score.sh Qwen1.5-72B-Chat gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gemma-2b-it gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gemma-7b-it gpt-4-turbo-2024-04-09


# bash evaluation/run_eval_v2_batch.score.sh Llama-2-7b-chat-hf gpt-4-turbo-2024-04-09 
# bash evaluation/run_eval_v2_batch.score.sh Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09

#### all of them 
# bash evaluation/run_eval_v2_batch.score.sh claude-3-opus-20240229 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh claude-3-sonnet-20240229 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh dbrx-instruct@together gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gemma-2b-it gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gemma-7b-it gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gpt-3.5-turbo-0125 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh gpt-4o-2024-05-13 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Llama-2-7b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Llama-2-13b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Meta-Llama-3-8B-Instruct gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Meta-Llama-3-70B-Instruct gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Mistral-7B-Instruct-v0.2 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh mistral-large-2402 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Mixtral-8x7B-Instruct-v0.1 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Nous-Hermes-2-Mixtral-8x7B-DPO gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Qwen1.5-7B-Chat@together gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Qwen1.5-72B-Chat gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Starling-LM-7B-beta gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh tulu-2-dpo-70b gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Yi-1.5-34B-Chat gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Yi-1.5-9B-Chat gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.score.sh Yi-1.5-6B-Chat gpt-4-turbo-2024-04-09 




 
# bash evaluation/run_eval_v2_batch.score.sh claude-3-opus-20240229 gpt-4-turbo-2024-04-09 claude-3-opus-20240229 

 