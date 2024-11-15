model_name=$1 # model to test 
# by default use "gpt-4-0125-preview" as gpt_eval_name
gpt_eval_name=${2:-"gpt-4o-2024-05-13"} # evaluator name  # gpt-4-0125-preview
num_shards=${3:-8} # shards 

total_ex=1024


eval_template="evaluation/eval_template.score.v2.md"
eval_folder="eval_results/v2.0625/score.v2/eval=${gpt_eval_name}/"
echo "Evaluating $model_name using $gpt_eval_name with $eval_template"
mkdir -p $eval_folder 



# Decide the shard size dynamically based on $num_shards and the total number 1024
if [ "$num_shards" -eq 1 ]; then
    eval_file="${eval_folder}/${model_name}.json"
    python src/eval.py \
        --action eval \
        --model $gpt_eval_name \
        --mode score \
        --eval_template $eval_template \
        --target_model_name $model_name \
        --max_words_to_eval -1 \
        --eval_output_file $eval_file 
else
    echo "Using $num_shards shards"
    shard_size=$(($total_ex / $num_shards))
    echo "Shard size: $shard_size"
    start_gpu=0 # not used 
    for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $num_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
        eval_file="${eval_folder}/${model_name}.$start-$end.json"
        echo "Evaluating $model_name from $start to $end"
        python src/eval.py \
            --action eval \
            --model $gpt_eval_name \
            --mode score \
            --eval_template $eval_template \
            --target_model_name $model_name \
            --max_words_to_eval -1 \
            --eval_output_file $eval_file \
            --start_idx $start --end_idx $end &
    done 
    # Wait for all background processes to finish
    wait

    # # Run the merge results script after all evaluation scripts have completed
    python src/merge_results.py $eval_folder $model_name
fi

 

