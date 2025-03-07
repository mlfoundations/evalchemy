#!/bin/bash
#SBATCH --nodes=64            
#SBATCH --ntasks=256           # 4 tasks per node - one per GPU
#SBATCH --cpus-per-task=8      # max
#SBATCH --gres=gpu:4           # use 4 GPU per node - max
#SBATCH --time=01:00:00        # run for 1 hour
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --account=p_finetuning

module load release/24.04  GCCcore/12.3.0
module load CUDA/12.1.1
module load NCCL/2.18.3-CUDA-12.1.1

export PATH="$HOME/miniconda3/condabin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate evalchemy

export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export HF_HOME="$WORK/hub"

cd $WORK/evalchemy
export $(grep -v '^#' .env | xargs) # ensure you have your env variables in .env file

# Login to Hugging Face
huggingface-cli login --token $HF_TOKEN

# CONFIG_FILE=$1
# Later can get these as arguments (like $1)
export TASK="REASONING"
export DATASET="mlfoundations-dev/${TASK}_evalchemy"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# GLOBAL_SIZE can be even larger (if you are running on different clusters or jobs e.g.)
export GLOBAL_SIZE=$SLURM_NTASKS
export LOG_DIR=logs/${MODEL_NAME}/${TASK}/
mkdir -p $LOG_DIR
echo "Logs out to $LOG_DIR" # add > $LOG_DIR/log_rank$SLURM_PROCID.txt 2>&1 to the end within ''

srun bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; echo "GLOBAL_SIZE: ${GLOBAL_SIZE},  SLURM_LOCALID: ${SLURM_LOCALID}, SLURM_PROCID: ${SLURM_PROCID}, DATASET: ${DATASET}, MODEL: ${MODEL_NAME}"'
srun bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; python $WORK/evalchemy/eval/examples/external_inference_sharded_vllm.py --global_size ${GLOBAL_SIZE} --rank ${SLURM_PROCID} --repo_id ${DATASET} --model_name ${MODEL_NAME}'