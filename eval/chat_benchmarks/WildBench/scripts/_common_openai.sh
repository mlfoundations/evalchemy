model_name=$1
model_pretty_name=$2
n_shards=$3 

TEMP=0; TOP_P=1.0; MAX_TOKENS=4096; 
batch_size=1;
# gpu="0,1,2,3"; num_gpus=4; 

CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/wild_bench_v2/"



# If the n_shards is 1, then we can directly run the model
# else, use  Data-parallellism
if [ $n_shards -eq 1 ]; then
    gpu="0,1,2,3"; num_gpus=4; # change the number of gpus to your preference
    echo "tsp = 1"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --engine openai \
        --data_name wild_bench \
        --model_name $model_name \
        --model_pretty_name $model_pretty_name \
        --top_p $TOP_P --temperature $TEMP \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $output_dir/  

elif [ $n_shards -gt 1 ]; then
    TOTAL_EXAMPLE=1024
    echo "Using Data-parallelism"
    start_gpu=0
    num_gpus=1
    shard_size=$((TOTAL_EXAMPLE/n_shards))
    shards_dir="${output_dir}/tmp_${model_pretty_name}"
    for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do

        CUDA_VISIBLE_DEVICES=$gpu \
        python src/unified_infer.py \
            --engine openai \
            --start_index $start --end_index $end \
            --data_name wild_bench \
            --model_name $model_name \
            --model_pretty_name $model_pretty_name \
            --top_p $TOP_P --temperature $TEMP \
            --batch_size $batch_size --max_tokens $MAX_TOKENS \
            --output_folder $shards_dir/ \
              &
    done 
    wait 
    python src/merge_results.py $shards_dir/ $model_pretty_name
    cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
else
    echo "Invalid n_shards"
    exit
fi

