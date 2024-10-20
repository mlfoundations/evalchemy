model_name=$1
model_pretty_name=$2
n_shards=$3 

TEMP=0; TOP_P=1.0; MAX_TOKENS=4096; 
batch_size=1;


CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/wild_bench_v2/"



# If the n_shards is 1, then we can directly run the model
# else, use  Data-parallellism
if [ $n_shards -eq 1 ]; then
    # gpu="0,1,2,3"; num_gpus=4; # change the number of gpus to your preference
    # decide the number gpus automatically from cuda 
    num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    # gpu= # from 0 to the last gpu id
    gpu=$(seq -s, 0 $((num_gpus - 1)))

    echo "n_shards = 1; num_gpus = $num_gpus; gpu = $gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --data_name wild_bench \
        --model_name $model_name \
        --use_hf_conv_template --use_imend_stop \
        --download_dir $CACHE_DIR \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
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
            --start_index $start --end_index $end \
            --data_name wild_bench \
            --model_name $model_name \
            --use_hf_conv_template --use_imend_stop \
            --download_dir $CACHE_DIR \
            --tensor_parallel_size $num_gpus \
            --dtype bfloat16 \
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

