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

# MAIN DIRECTORIES
export DCFT="$WORK/DCFT_shared"
export EVALCHEMY="$DCFT/evalchemy"
export HF_HUB_CACHE="$DCFT/hub"

# CONDA
SHARED_MAMBA_INSTALL=$DCFT/mamba
SHARED_EVALCHEMY_GPU_ENV=$EVALCHEMY/env/cu121-evalchemy
source ${SHARED_MAMBA_INSTALL}/bin/activate ${SHARED_EVALCHEMY_GPU_ENV}

# SHARDED INFERENCE ARGUMENTS
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
# export MODEL_NAME="$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export INPUT_DATASET="$HF_HUB_CACHE/datasets--mlfoundations-dev--evalset_2870"
export OUTPUT_DIR="$EVALCHEMY/results/${MODEL_NAME##*--}_${INPUT_DATASET##*--}"

# RUN SHARDED INFERENCE
srun echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DIR: ${OUTPUT_DIR}"
srun python $EVALCHEMY/eval/examples/external_inference_sharded_vllm.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --model_name ${MODEL_NAME} --output_dir ${OUTPUT_DIR} --offline --no_upload