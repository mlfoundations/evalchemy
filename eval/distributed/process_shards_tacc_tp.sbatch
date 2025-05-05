#!/bin/bash
#SBATCH --array=0
#SBATCH --nodes=4            
#SBATCH --ntasks-per-node 1         
#SBATCH --time=01:00:00     
#SBATCH --cpus-per-task=72
#SBATCH -p gh
#SBATCH --job-name=process_shards
#SBATCH --exclude=c610-021,c611-011
#SBATCH --account CCR24067


# MODULES
module load cuda/12.4 nccl/12.4

# ENVIRONMENT VARIABLES - EVALCHEMY, HF_HUB_CACHE, and EVALCHEMY_ACTIVATE_ENV
source /scratch/08002/gsmyrnis/dcft_shared/dcft_private/hpc/dotenv/tacc.env

# CONDA
$EVALCHEMY_ACTIVATE_ENV

# LAUNCH RAY CLUSTER
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 72 --num-gpus 1 --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus 72 --num-gpus 1 --block &
    sleep 5
done


# SHARDED INFERENCE ARGUMENTS
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
export MODEL_NAME="$HF_HUB_CACHE/models--open-thoughts--OpenThinker-7B/snapshots/5a931fd3fa8618acda2da8eaec4a3f10ee009739"
export INPUT_DATASET="$HF_HUB_CACHE/datasets--mlfoundations-dev--evalset_2870"
export OUTPUT_DATASET="$EVALCHEMY/results/${MODEL_NAME##*--}_${INPUT_DATASET##*--}"

# RUN SHARDED INFERENCE
srun echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DATASET: ${OUTPUT_DATASET}"
srun python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --model_name ${MODEL_NAME} --output_dataset ${OUTPUT_DATASET} --tp 4
