#!/bin/bash
#SBATCH --nodes=1            
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:1           
#SBATCH --time=01:00:00        
#SBATCH --array=0-1                   # Number of shards (independent tasks with 1 GPU each)
#SBATCH --account=EUHPC_E03_068
#SBATCH --partition=boost_usr_prod

# MODULES
module load cuda/12.1
module load gcc/12.2.0
module load nccl

# MAIN EVALCHEMY DIRECTORY
export EVALCHEMY="$WORK/DCFT_shared/evalchemy"

# CONDA
source $EVALCHEMY/env/miniconda/bin/activate 

# HUGGINGFACE
export HF_HUB_CACHE="$WORK/DCFT_shared/hub"
huggingface-cli login --token $HF_TOKEN

# LOAD IN ENVIRONMENT VARIABLES FROM .ENV FILE
export $(grep -v '^#' $EVALCHEMY/.env | xargs)

# SHARDED INFERENCE ARGUEMENTS
export TASK="AIME24"
export INPUT_DATASET="mlfoundations-dev/evalset_2870" #AIME24
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export MODEL_NAME_SHORT=${MODEL_NAME##*/}
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
export OUTPUT_DATASET="mlfoundations-dev/${TASK}_evalchemy_${GLOBAL_SIZE}shards_${MODEL_NAME_SHORT}"
export OUTPUT_DIR="$EVALCHEMY/results/${TASK}_${MODEL_NAME_SHORT}"

# Create output directory
mkdir -p $OUTPUT_DIR

# SRUN TASKS
srun bash -c 'echo "GLOBAL_SIZE: ${GLOBAL_SIZE},  RANK: ${RANK}, INPUT_DATASET: ${INPUT_DATASET}, OUTPUT_DIR: ${OUTPUT_DIR}, MODEL: ${MODEL_NAME}"'
srun bash -c 'python $EVALCHEMY/eval/examples/external_inference_sharded_vllm.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --output_dataset ${OUTPUT_DATASET} --model_name ${MODEL_NAME} --no_upload --output_dir ${OUTPUT_DIR}'