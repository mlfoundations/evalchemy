#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=12
#SBATCH --account=westai0007
#SBATCH --partition=dc-hwai
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=dcft-slurm-notifs-aaaap7wt363mcsgryaejj2o6dm@dogs-and-ml.slack.com

# ENVIRONMENT VARIABLES
source /p/project1/laionize/dcft/dcft_private/hpc/dotenv/jureca.env

# CONDA
echo "Loading conda: $EVALCHEMY_ACTIVATE_ENV"
$EVALCHEMY_ACTIVATE_ENV

# DOWNLOAD MODEL AND DATASET
MODEL_NAME={model_name}
INPUT_DATASET={input_dataset}
OUTPUT_DATASET={output_dataset}

# RUN SHARDED INFERENCE
srun --output={logs_dir}/%x_%j_%t.out bash -c 'echo -e "GLOBAL_SIZE: ${SLURM_STEP_NUM_TASKS}\nRANK: ${SLURM_PROCID}\nMODEL: '$MODEL_NAME'\nINPUT_DATASET: '$INPUT_DATASET'\nOUTPUT_DATASET: '$OUTPUT_DATASET'"'
srun --output={logs_dir}/%x_%j_%t.out bash -c 'CUDA_VISIBLE_DEVICES=${SLURM_LOCALID} python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${SLURM_STEP_NUM_TASKS} --rank ${SLURM_PROCID} --input_dataset '${INPUT_DATASET}' --model_name '${MODEL_NAME}' --output_dataset '${OUTPUT_DATASET}''