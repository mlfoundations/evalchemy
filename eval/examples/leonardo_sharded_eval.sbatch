#!/bin/bash
#SBATCH --array=0-1
#SBATCH --nodes=1            
#SBATCH --ntasks=1           
#SBATCH --gres=gpu:1           
#SBATCH --time=01:00:00        
#SBATCH --account=EUHPC_E03_068
#SBATCH --partition=boost_usr_prod

# MODULES
module load cuda/12.1
module load gcc/12.2.0
module load nccl

# DIRECTORIES
export DCFT="$WORK/DCFT_shared"
export EVALCHEMY="$DCFT/evalchemy"
export HF_HUB_CACHE="$DCFT/hub"

# CONDA
source $EVALCHEMY/env/miniconda/bin/activate 

# SHARDED INFERENCE ARGUEMENTS
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
export MODEL_NAME="$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct"
export INPUT_DATASET="$HF_HUB_CACHE/datasets--mlfoundations-dev--evalset_2870"
export OUTPUT_DIR="$EVALCHEMY/results/${MODEL_NAME##*--}_${INPUT_DATASET##*--}"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run sharded inference 
srun bash -c 'echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DIR: ${OUTPUT_DIR}"'
srun bash -c 'python $EVALCHEMY/eval/examples/external_inference_sharded_vllm.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --model_name ${MODEL_NAME} --no_upload --output_dir ${OUTPUT_DIR}'