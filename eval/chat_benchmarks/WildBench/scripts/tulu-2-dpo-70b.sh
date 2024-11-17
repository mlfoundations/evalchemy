model_name="allenai/tulu-2-dpo-70b"
model_pretty_name="tulu-2-dpo-70b"
TEMP=0; TOP_P=1.0; MAX_TOKENS=4096; 
gpu="0,1,2,3"; num_gpus=4; batch_size=4;

CACHE_DIR=${HF_HOME:-"default"}
output_dir="result_dirs/wild_bench_v2/"

CUDA_VISIBLE_DEVICES=$gpu \
python src/unified_infer.py \
    --data_name wild_bench \
    --model_name $model_name \
    --download_dir $CACHE_DIR \
    --tensor_parallel_size $num_gpus \
    --dtype bfloat16 \
    --top_p $TOP_P --temperature $TEMP \
    --batch_size $batch_size --max_tokens $MAX_TOKENS \
    --output_folder $output_dir/ \
      