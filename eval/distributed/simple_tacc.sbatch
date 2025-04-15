#!/bin/bash
#SBATCH --nodes={num_shards}
#SBATCH --tasks={num_shards}
#SBATCH --time={time_limit}   
#SBATCH --cpus-per-task=72
#SBATCH --partition=gh
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --exclude=c610-021,c611-011
#SBATCH --account CCR24067

# EXIT ON ERROR
set -e

# MODULES
module load cuda/12.4 nccl/12.4

# CONDA
source /work/08134/negin/anaconda3/bin/activate evalchemy

# DCFT DIRECTORIES
export DCFT="/work/08134/negin/ls6/shared_env/"
export EVALCHEMY="$DCFT/evalchemy"
export HF_HUB_CACHE="$DCFT/hub"

# DOWNLOAD MODEL AND DATASET
MODEL_NAME={model_name}
INPUT_DATASET={input_dataset}
OUTPUT_DATASET={output_dataset}
srun --nodes=1 --tasks=1 huggingface-cli download $MODEL_NAME --cache-dir $HF_HUB_CACHE
srun --nodes=1 --tasks=1 huggingface-cli download $INPUT_DATASET --cache-dir $HF_HUB_CACHE --repo-type dataset

# RUN SHARDED INFERENCE
# # Create a SLURM_NODEID equivalent
# export NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | sort)
# export CURRENT_HOST=$(hostname -s)  # Use short hostname without domain
# export SLURM_NODEID=$(echo "$NODELIST" | grep -n "$CURRENT_HOST" | cut -d':' -f1)
# export SLURM_NODEID=$((SLURM_NODEID - 1))  # Convert to 0-based indexing

# # Debug output
# echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
# echo "NODELIST: $NODELIST"
# echo "CURRENT_HOST: $CURRENT_HOST" 
# echo "SLURM_NODEID: $SLURM_NODEID"

srun --nodes=1 echo -e "GLOBAL_SIZE: ${SLURM_JOB_NUM_NODES}\nRANK: ${SLURM_PROCID}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DATASET: ${OUTPUT_DATASET}"
# srun python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${SLURM_JOB_NUM_NODES} --rank ${SLURM_PROCID} --input_dataset ${INPUT_DATASET} --model_name ${MODEL_NAME} --output_dataset ${OUTPUT_DATASET} --upload