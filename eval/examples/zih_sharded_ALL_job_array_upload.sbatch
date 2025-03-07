#!/bin/bash
#SBATCH --nodes=1            
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:1           
#SBATCH --time=01:00:00        
#SBATCH --mem=64G
#SBATCH --array=0-127 # Create 128 independent tasks (or however many idle gpus are there!)
#SBATCH --account=p_finetuning

# MODULES
module load release/24.04  GCCcore/12.3.0
module load CUDA/12.1.1
module load NCCL/2.18.3-CUDA-12.1.1

# CONDA
export PATH="$HOME/miniconda3/condabin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate evalchemy

# CUDA PATHS
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

# HUGGINGFACE
export HF_HOME="$WORK/hub"
huggingface-cli login --token $HF_TOKEN

# LOAD IN ENVIRONMENT VARIABLES FROM .ENV FILE
cd $WORK/evalchemy
export $(grep -v '^#' .env | xargs)

# SHARDED INFERENCE ARGUEMENTS
export TASK="REASONING"
export INPUT_DATASET="mlfoundations-dev/${TASK}_evalchemy"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export MODEL_NAME_SHORT=${MODEL_NAME##*/}
export GLOBAL_SIZE=$SLURM_ARRAY_TASK_COUNT
export RANK=$SLURM_ARRAY_TASK_ID
export OUTPUT_DATASET="mlfoundations-dev/${TASK}_evalchemy_${GLOBAL_SIZE}shards_${MODEL_NAME_SHORT}"
export OUTPUT_DIR="$WORK/evalchemy_results/${TASK}/${MODEL_NAME_SHORT}"

# Create output directory
mkdir -p $OUTPUT_DIR

# SRUN TASKS
srun bash -c 'echo "GLOBAL_SIZE: ${GLOBAL_SIZE},  RANK: ${RANK}, INPUT_DATASET: ${INPUT_DATASET}, OUTPUT_DIR: ${OUTPUT_DIR}, MODEL: ${MODEL_NAME}"'
srun bash -c 'python $WORK/evalchemy/eval/examples/external_inference_sharded_vllm.py --global_size ${GLOBAL_SIZE} --rank ${RANK} --input_dataset ${INPUT_DATASET} --output_dataset ${OUTPUT_DATASET} --model_name ${MODEL_NAME} --no_upload --output_dir ${OUTPUT_DIR}'

# Where does a command below here run? Which compute node? 
# Should we do the LCB run here?
# Note: After all jobs are complete, you can upload all shards to HuggingFace Hub with:
# huggingface-cli repo create ${OUTPUT_DATASET} --type dataset
# huggingface-cli upload ${OUTPUT_DATASET} ${OUTPUT_DIR} --repo-type dataset