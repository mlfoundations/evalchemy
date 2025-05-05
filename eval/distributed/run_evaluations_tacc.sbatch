#!/bin/bash
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1         
#SBATCH --time=01:00:00        
#SBATCH --cpus-per-task=144
#SBATCH -p gg
#SBATCH --job-name=run_evaluations
#SBATCH --account CCR24067

# MODULES
module load cuda/12.4 nccl/12.4

# ENVIRONMENT VARIABLES - EVALCHEMY, HF_HUB_CACHE, and EVALCHEMY_ACTIVATE_ENV
source /scratch/08002/gsmyrnis/dcft_shared/dcft_private/hpc/dotenv/tacc.env

# CONDA
$EVALCHEMY_ACTIVATE_ENV

# EVAL SCORING COMMAND
export EVAL_COMMAND="python -m eval.eval --model precomputed_hf --model_args "repo_id=mlfoundations-dev/Qwen2.5-7B-Instruct_1744582706_eval_288a",model="Qwen/Qwen2.5-7B-Instruct" --tasks AIME24,AMC23,MATH500,AIME25,GPQADiamond,MMLUPro,LiveCodeBench,CodeElo --output_path logs --use_database"

# RUN EVAL SCORING
srun echo -e "EVAL_COMMAND: ${EVAL_COMMAND}"
srun ${EVAL_COMMAND}
