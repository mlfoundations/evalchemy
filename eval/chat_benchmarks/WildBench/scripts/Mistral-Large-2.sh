# export MISTRAL_API_KEY=your_mistral_api_key
model_name="mistral/mistral-large-2407"
model_pretty_name="Mistral-Large-2"
output_dir="result_dirs/wild_bench_v2/"
TEMP=0; TOP_P=1.0; MAX_TOKENS=4096;

# shard_size should be 1024 // n_shards
n_shards=8
shard_size=128
start_gpu=0
shards_dir="${output_dir}/tmp_${model_pretty_name}"
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    python src/unified_infer.py \
        --data_name wild_bench \
        --start_index $start --end_index $end \
        --engine mistral \
        --model_name $model_name \
        --top_p $TOP_P --temperature $TEMP \
        --max_tokens $MAX_TOKENS \
        --output_folder $shards_dir/ \
         &
done 
wait 
python src/merge_results.py $shards_dir/ "mistral-large-2407"
cp $shards_dir/mistral-large-2407.json $output_dir/${model_pretty_name}.json
