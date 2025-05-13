#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=96
#SBATCH --account=rwth1775
#SBATCH --partition=c23g
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x/%j.out
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=dcft-slurm-notifs-aaaap7wt363mcsgryaejj2o6dm@dogs-and-ml.slack.com

# EXIT ON ERROR
set -e

# CUDA
module load CUDA/12.6.0
module load imkl/2023.2.0

# ENVIRONMENT VARIABLES - EVALCHEMY, HF_HUB_CACHE, and EVALCHEMY_ACTIVATE_ENV
export DCFT=/rwthfs/rz/cluster/hpcwork/rwth1775/dcft
source $DCFT/dcft_private/hpc/dotenv/claix.env
source $DCFT/dcft_private/database/access.env

# CONDA
echo "Loading conda: $EVALCHEMY_ACTIVATE_ENV"
$EVALCHEMY_ACTIVATE_ENV

# DOWNLOAD MODEL AND DATASET
MODEL_NAME={model_name}
INPUT_DATASET={input_dataset}
OUTPUT_DATASET={output_dataset}
GLOBAL_SIZE=$((${SLURM_JOB_NUM_NODES}*4))
srun --nodes=1 --ntasks=1 huggingface-cli download $MODEL_NAME --cache-dir $HF_HUB_CACHE
srun --nodes=1 --ntasks=1 huggingface-cli download $INPUT_DATASET --cache-dir $HF_HUB_CACHE --repo-type dataset

# RUN SHARDED INFERENCE (without wait=0 infinite wait it will kill tasks after the first task finishes)
# The default is 5 seconds on claix
# ❯ scontrol show config | grep WaitTime
# WaitTime                = 5 sec
srun --nodes={num_nodes} --wait=0 --ntasks=$GLOBAL_SIZE --ntasks-per-node=4 --cpus-per-task=24 --output={logs_dir}/%x/%j_%t.out bash -c 'echo -e "GLOBAL_SIZE: ${SLURM_STEP_NUM_TASKS}\nRANK: ${SLURM_PROCID}\nMODEL: '$MODEL_NAME'\nINPUT_DATASET: '$INPUT_DATASET'\nOUTPUT_DATASET: '$OUTPUT_DATASET'"'
srun --nodes={num_nodes} --wait=0 --ntasks=$GLOBAL_SIZE --ntasks-per-node=4 --cpus-per-task=24 --output={logs_dir}/%x/%j_%t.out bash -c 'CUDA_VISIBLE_DEVICES=${SLURM_LOCALID} python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${SLURM_STEP_NUM_TASKS} --rank ${SLURM_PROCID} --input_dataset '${INPUT_DATASET}' --model_name '${MODEL_NAME}' --output_dataset '${OUTPUT_DATASET}' --upload'

# COMPUTE SCORES
srun --nodes=1 --ntasks=1 --mem=0 --cpus-per-task=96 python -m eval.eval --model precomputed_hf --model_args "repo_id={output_dataset}",model="{model_name}" --tasks {tasks_str} --output_path logs --use_database