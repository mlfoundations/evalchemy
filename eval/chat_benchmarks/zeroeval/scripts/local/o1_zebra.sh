

# bash zero_eval_api.sh -f openai -d zebra-grid -m openai/o1-mini-2024-09-12 -p o1-mini-2024-09-12 -s 1

DATA_NAME="zebra-grid"
engine_name="openai"
model_name="openai/o1-preview-2024-09-12"
model_pretty_name="o1-preview-2024-09-12"

run_name="default"
TEMP=0
TOP_P=1.0
rp=1.0
MAX_TOKENS=4096
batch_size=1

# CACHE_FILE="result_dirs/${DATA_NAME}/${model_pretty_name}.legacy.json"
# echo "CACHE_FILE=$CACHE_FILE"

n_shards=8
shard_id=6
output_dir="result_dirs/${DATA_NAME}/tmp_${model_pretty_name}/"

python src/unified_infer.py \
    --data_name $DATA_NAME \
    --engine $engine_name \
    --model_name $model_name \
    --model_pretty_name $model_pretty_name \
    --run_name $run_name \
    --num_shards $n_shards \
    --shard_id $shard_id \
    --top_p $TOP_P --temperature $TEMP --repetition_penalty $rp \
    --batch_size $batch_size --max_tokens $MAX_TOKENS \
    --output_folder $output_dir/


    # --cache_filepath $CACHE_FILE \
    