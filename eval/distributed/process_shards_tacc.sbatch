#!/bin/bash
#SBATCH --array=0
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node 1         
#SBATCH --time=01:00:00        
#SBATCH --cpus-per-task=8 
#SBATCH --account=EUHPC_E03_068
#SBATCH --partition=boost_usr_prod
#SBATCH -p gh
#SBATCH --job-name=process_shards
#SBATCH --exclude=c610-021,c611-011
#SBATCH --account CCR24067


# MODULES
module load cuda/12.4 nccl/12.4

# MAIN DIRECTORIES
export DCFT="/work/08134/negin/ls6"
export EVALCHEMY="$DCFT/evalchemy"
export HF_HUB_CACHE="$SCRATCH/hf_home"

# CONDA
export PATH="/work/08134/negin/anaconda3/condabin/:$PATH"
source /work/08134/negin/anaconda3/etc/profile.d/conda.sh
conda activate evalchemy

# SHARDED INFERENCE ARGUMENTS
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
export MODEL_NAME="$HF_HUB_CACHE/models--open-thoughts--OpenThinker-7B/snapshots/5a931fd3fa8618acda2da8eaec4a3f10ee009739"
export INPUT_DATASET="$HF_HUB_CACHE/datasets--mlfoundations-dev--evalset_2870"
export OUTPUT_DATASET="$EVALCHEMY/results/${MODEL_NAME##*--}_${INPUT_DATASET##*--}"

# RUN SHARDED INFERENCE
srun echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DATASET: ${OUTPUT_DATASET}"
srun python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --model_name ${MODEL_NAME} --output_dataset ${OUTPUT_DATASET}
