# DATA_NAME=$1
# engine_name=$2
# model_name=$3
# model_pretty_name=$4
# n_shards=$5
# # default cot to be True 
# cot=${6:-True}
# TEMP=0; TOP_P=1.0; 

# Initialize default values
DATA_NAME=""
model_name=""
model_pretty_name=""
n_shards=1
run_name="default"
TEMP=0
TOP_P=1.0
rp=1.0
engine_name="openai"
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
  echo "Usage: $0 -d DATA_NAME -m model_name -p model_pretty_name -s n_shards [-r run_name] [-t TEMP] [-o TOP_P] [-e rp] [-f engine_name]"
  exit 1
fi





batch_size=4; 
CACHE_DIR=${HF_HOME:-"default"}
# output_dir="result_dirs/${DATA_NAME}/cot=${cot}/" 

if [ "$run_name" = "default" ]; then
    output_dir="result_dirs/${DATA_NAME}/" 
else
    output_dir="result_dirs/${DATA_NAME}/${run_name}/" 
fi


# If the n_shards is 1, then we can directly run the model
# else, use  Data-parallellism
if [ $n_shards -eq 1 ]; then
    echo "n_shards = 1"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --data_name $DATA_NAME \
        --engine $engine_name \
        --model_name $model_name \
        --model_pretty_name $model_pretty_name \
        --run_name $run_name \
        --top_p $TOP_P --temperature $TEMP --repetition_penalty $rp \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $output_dir/  

elif [ $n_shards -gt 1 ]; then
    echo "Using Data-parallelism"
    start_gpu=0 
    shards_dir="${output_dir}/tmp_${model_pretty_name}"
    for ((shard_id = 0; shard_id < $n_shards; shard_id++, gpu++)); do
        python src/unified_infer.py \
            --num_shards $n_shards \
            --shard_id $shard_id \
            --data_name $DATA_NAME \
            --engine $engine_name \
            --model_name $model_name \
            --run_name $run_name \
            --model_pretty_name $model_pretty_name \
            --top_p $TOP_P --temperature $TEMP --repetition_penalty $rp \
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

