#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 # MinTRES is gres/gpu=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --account=p_finetuning

# CONDA
export DCFT=/data/horse/ws/ryma833h-DCFT_Shared
export EVALCHEMY=/data/horse/ws/ryma833h-DCFT_Shared/evalchemy/
export PATH="$DCFT/miniconda3/condabin:$PATH"
source $DCFT/miniconda3/etc/profile.d/conda.sh
conda activate evalchemy

# EVAL SCORING COMMAND
export EVAL_COMMAND="python -m eval.eval --model precomputed_hf --model_args "repo_id=mlfoundations-dev/Qwen2.5-7B-Instruct_1744582706_eval_288a",model="Qwen/Qwen2.5-7B-Instruct" --tasks AIME24,AMC23,MATH500,AIME25,GPQADiamond,MMLUPro,LiveCodeBench,CodeElo --output_path logs --use_database"

# RUN EVAL SCORING
srun echo -e "EVAL_COMMAND: ${EVAL_COMMAND}"
srun ${EVAL_COMMAND}
