#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --time={time_limit}   
#SBATCH --cpus-per-task=72
#SBATCH --partition=gh
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --exclude=c610-021,c611-011
#SBATCH --account CCR24067
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=dcft-slurm-notifs-aaaap7wt363mcsgryaejj2o6dm@dogs-and-ml.slack.com

# EXIT ON ERROR
set -e

# MODULES
module load cuda/12.4 nccl/12.4

# ENVIRONMENT VARIABLES - EVALCHEMY, HF_HUB_CACHE, and EVALCHEMY_ACTIVATE_ENV
source /scratch/08002/gsmyrnis/dcft_shared/dcft_private/hpc/dotenv/tacc.env

# CONDA
$EVALCHEMY_ACTIVATE_ENV

# DOWNLOAD MODEL AND DATASET
MODEL_NAME={model_name}
INPUT_DATASET={input_dataset}
OUTPUT_DATASET={output_dataset}
srun --nodes=1 huggingface-cli download $MODEL_NAME --cache-dir $HF_HUB_CACHE
srun --nodes=1 huggingface-cli download $INPUT_DATASET --cache-dir $HF_HUB_CACHE --repo-type dataset

# RUN SHARDED INFERENCE
srun --output={logs_dir}/%x_%j_%n.out bash -c 'echo -e "GLOBAL_SIZE: ${SLURM_JOB_NUM_NODES}\nRANK: ${SLURM_NODEID}\nMODEL: '$MODEL_NAME'\nINPUT_DATASET: '$INPUT_DATASET'\nOUTPUT_DATASET: '$OUTPUT_DATASET'"'
srun --output={logs_dir}/%x_%j_%n.out bash -c 'python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${SLURM_JOB_NUM_NODES} --rank ${SLURM_NODEID} --input_dataset '${INPUT_DATASET}' --model_name '${MODEL_NAME}' --output_dataset '${OUTPUT_DATASET}' --upload'

# COMPUTE SCORES
srun --nodes=1 python -m eval.eval --model precomputed_hf --model_args "repo_id={output_dataset}",model="{model_name}" --tasks {tasks_str} --output_path logs --use_database
