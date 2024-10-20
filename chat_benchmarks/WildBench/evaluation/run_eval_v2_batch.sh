model_name=$1 # model to test 
# by default use gpt-3.5-turbo-0125 as ref_name
ref_name=${2:-"gpt-3.5-turbo-0125"} # model to compare 
# by default use "gpt-4-0125-preview" as gpt_eval_name
gpt_eval_name=${3:-"gpt-4-turbo-2024-04-09"} # evaluator name  # gpt-4-0125-preview




eval_template="evaluation/eval_template.pairwise.v2.md"
eval_folder="eval_results/v2.0522/pairwise.v2/eval=${gpt_eval_name}/ref=${ref_name}/"
echo "Evaluating $model_name vs $ref_name using $gpt_eval_name with $eval_template"


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
    --max_words_to_eval 1000 \
    --mode pairwise \
    --eval_template $eval_template \
    --target_model_name $model_name \
    --ref_model_name $ref_name \
    --eval_output_file $eval_file 

echo "Batch results saved to $eval_file"

## V2.0522

# test_model, ref_model, eval_model

# bash evaluation/run_eval_v2_batch.sh claude-3-opus-20240229 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-70B-Instruct gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-8B-Instruct gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh claude-3-sonnet-20240229 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4o-2024-05-13 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09





### All vs GPT-4-turbo
# bash evaluation/run_eval_v2_batch.sh claude-3-haiku-20240307 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh claude-3-sonnet-20240229 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh claude-3-opus-20240229 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh dbrx-instruct@together gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gemma-2b-it gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gemma-7b-it gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-3.5-turbo-0125 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4o-2024-05-13 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-7b-chat-hf gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-13b-chat-hf gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-8B-Instruct gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-70B-Instruct gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Mistral-7B-Instruct-v0.2 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh mistral-large-2402 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Mixtral-8x7B-Instruct-v0.1 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Nous-Hermes-2-Mixtral-8x7B-DPO gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Qwen1.5-7B-Chat@together gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09 
# bash evaluation/run_eval_v2_batch.sh Starling-LM-7B-beta gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh tulu-2-dpo-70b gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09

# bash evaluation/run_eval_v2_batch.sh Yi-1.5-34B-Chat gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-9B-Chat gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-6B-Chat gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09

# bash evaluation/run_eval_v2_batch.sh gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09 gpt-4-turbo-2024-04-09

### All vs Llama-2-70b-chat-hf

# bash evaluation/run_eval_v2_batch.sh claude-3-opus-20240229 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh claude-3-haiku-20240307 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh claude-3-sonnet-20240229 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh dbrx-instruct@together Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gemma-2b-it Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gemma-7b-it Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-3.5-turbo-0125 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4-turbo-2024-04-09 Llama-2-70b-chat-hf  gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4o-2024-05-13 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-7b-chat-hf Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-8B-Instruct Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-70B-Instruct Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Mistral-7B-Instruct-v0.2 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh mistral-large-2402 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Mixtral-8x7B-Instruct-v0.1 Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Nous-Hermes-2-Mixtral-8x7B-DPO Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Qwen1.5-7B-Chat@together Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Qwen1.5-72B-Chat Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Starling-LM-7B-beta Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh tulu-2-dpo-70b Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-34B-Chat Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-9B-Chat Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-6B-Chat Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh command-r Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4-0125-preview Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09


### ALl vs Haiku 

# bash evaluation/run_eval_v2_batch.sh claude-3-opus-20240229 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-70b-chat-hf claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh claude-3-sonnet-20240229 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh dbrx-instruct@together claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gemma-2b-it claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gemma-7b-it claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-3.5-turbo-0125 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4-turbo-2024-04-09 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4o-2024-05-13 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-7b-chat-hf claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-8B-Instruct claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Meta-Llama-3-70B-Instruct claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Mistral-7B-Instruct-v0.2 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh mistral-large-2402 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Mixtral-8x7B-Instruct-v0.1 claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Nous-Hermes-2-Mixtral-8x7B-DPO claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Qwen1.5-7B-Chat@together claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Qwen1.5-72B-Chat claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Starling-LM-7B-beta claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh tulu-2-dpo-70b claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-34B-Chat claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-9B-Chat claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Yi-1.5-6B-Chat claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh command-r claude-3-haiku-20240307 gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh gpt-4-0125-preview claude-3-haiku-20240307 gpt-4-turbo-2024-04-09


 







# bash evaluation/run_eval_v2_batch.sh Llama-2-13b-chat-hf Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09
# bash evaluation/run_eval_v2_batch.sh Llama-2-70b-chat-hf Llama-2-70b-chat-hf gpt-4-turbo-2024-04-09

# bash evaluation/run_eval_v2_batch.sh claude-3-opus-20240229 gpt-4-turbo-2024-04-09 claude-3-opus-20240229 


