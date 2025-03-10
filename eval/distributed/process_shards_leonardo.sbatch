#!/bin/bash
#SBATCH --array=0-1
#SBATCH --nodes=1            
#SBATCH --ntasks=1           
#SBATCH --gres=gpu:1           
#SBATCH --time=01:00:00        
#SBATCH --cpus-per-task=8 
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
EVALCHEMY_GPU_ENV=$EVALCHEMY/env/cu121-evalchemy
source $DCFT/mamba/bin/activate $EVALCHEMY_GPU_ENV

# SHARDED INFERENCE ARGUMENTS
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
export MODEL_NAME="$HF_HUB_CACHE/models--open-thoughts--OpenThinker-7B/snapshots/5a931fd3fa8618acda2da8eaec4a3f10ee009739"
export INPUT_DATASET="$HF_HUB_CACHE/datasets--mlfoundations-dev--evalset_2870"
export OUTPUT_DATASET="$EVALCHEMY/results/${MODEL_NAME##*--}_${INPUT_DATASET##*--}"

# RUN SHARDED INFERENCE
srun echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DATASET: ${OUTPUT_DATASET}"
srun python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --model_name ${MODEL_NAME} --output_dataset ${OUTPUT_DATASET}