#!/bin/bash


HF_HUB_ENABLE_HF_TRANSFER=1

# Initialize default values
DATA_NAME=""
model_name=""
model_pretty_name=""
n_shards=1
run_name="default"
TEMP=0
TOP_P=1.0
rp=1.0
engine_name="vllm"
batch_size=4

gpu_memory_utilization=0.95

MAX_TOKENS=4096; 

# Parse named arguments
while getopts ":d:m:p:s:r:t:o:e:f:b:x:" opt; do
  case $opt in
    d) DATA_NAME="$OPTARG"
    ;;
    m) model_name="$OPTARG"
    ;;
    p) model_pretty_name="$OPTARG"
    ;;
    s) n_shards="$OPTARG"
    ;;
    r) run_name="$OPTARG"
    ;;
    t) TEMP="$OPTARG"
    ;;
    o) TOP_P="$OPTARG"
    ;;
    e) rp="$OPTARG"
    ;;
    f) engine_name="$OPTARG"
    ;;
    b) batch_size="$OPTARG"
    ;;
    x) MAX_TOKENS="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Check if required arguments are provided
if [ -z "$DATA_NAME" ] || [ -z "$model_name" ] || [ -z "$model_pretty_name" ] || [ -z "$n_shards" ]; then
  echo "Usage: $0 -d DATA_NAME -m model_name -p model_pretty_name -s n_shards [-r run_name] [-t TEMP] [-o TOP_P] [-e rp] [-b batch_size]"
  exit 1
fi



CACHE_DIR=${HF_HOME:-"default"}
if [ "$run_name" = "default" ]; then
    output_dir="result_dirs/${DATA_NAME}/" 
else
    output_dir="result_dirs/${DATA_NAME}/${run_name}/" 
fi

# if model name contains "gemma-2" then use a different vllm infer backend
if [[ $model_name == *"gemma-2"* ]]; then
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    # if 27b in model name, then use 0.8 gpu memory utilization
    if [[ $model_name == *"27b"* ]]; then
        gpu_memory_utilization=0.8
        echo "Using 0.8 gpu memory utilization"
    fi 
fi




max_model_len=-1

# if model name contains "phi-3.5" then use a different gpu_memory_utilization
if [[ $model_name == *"Phi-3.5"* ]]; then
    gpu_memory_utilization=0.9
    max_model_len=4096
fi

if [[ $model_name == *"70B"* ]]; then
        gpu_memory_utilization=0.9
        max_model_len=4096
fi 

echo "Using ${gpu_memory_utilization} gpu memory utilization and max_model_len=${max_model_len}"



# If the n_shards is 1, then we can directly run the model
# else, use  Data-parallellism
if [ $n_shards -eq 1 ]; then
    # gpu="0,1,2,3"; num_gpus=4; # change the number of gpus to your preference
    # echo "n_shards = 1"
    num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    # gpu= # from 0 to the last gpu id
    gpu=$(seq -s, 0 $((num_gpus - 1)))

    echo "n_shards = 1; num_gpus = $num_gpus; gpu = $gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --engine $engine_name \
        --data_name $DATA_NAME \
        --model_name $model_name \
        --run_name $run_name \
        --gpu_memory_utilization $gpu_memory_utilization \
        --max_model_len $max_model_len \
        --use_hf_conv_template --use_imend_stop \
        --download_dir $CACHE_DIR \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --model_pretty_name $model_pretty_name \
        --top_p $TOP_P --temperature $TEMP \
        --repetition_penalty $rp \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $output_dir/  

elif [ $n_shards -gt 1 ]; then
    echo "Using Data-parallelism"
    start_gpu=0
    num_gpus=1 
    shards_dir="${output_dir}/tmp_${model_pretty_name}"
    for ((shard_id = 0, gpu = $start_gpu; shard_id < $n_shards; shard_id++, gpu++)); do
        CUDA_VISIBLE_DEVICES=$gpu \
        python src/unified_infer.py \
            --engine $engine_name \
            --num_shards $n_shards \
            --shard_id $shard_id \
            --data_name $DATA_NAME \
            --model_name $model_name \
            --run_name $run_name \
            --gpu_memory_utilization $gpu_memory_utilization \
            --max_model_len $max_model_len \
            --use_hf_conv_template --use_imend_stop \
            --download_dir $CACHE_DIR \
            --tensor_parallel_size $num_gpus \
            --dtype bfloat16 \
            --model_pretty_name $model_pretty_name \
            --top_p $TOP_P --temperature $TEMP \
            --repetition_penalty $rp \
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
 