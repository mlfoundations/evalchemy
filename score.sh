#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --time=01:00:00   
#SBATCH --cpus-per-task=72
#SBATCH --partition=gg
#SBATCH --job-name=eval_score
#SBATCH --output=logs/%x_%j.out
#SBATCH --account CCR24067
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=dcft-slurm-notifs-aaaap7wt363mcsgryaejj2o6dm@dogs-and-ml.slack.com


TAG="${1##*/}"               # strip everything before the last “/”
NEWNAME="score_${TAG}"
scontrol update JobId=$SLURM_JOB_ID JobName="$NEWNAME"
echo "Job name changed to: $NEWNAME"

# 1. PARSE CLI ARGUMENTS
if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch $0 <HF_DATASET_ID>";  exit 1;
fi
INPUT_DATASET="$1"
HASH="${INPUT_DATASET##*_eval_}"      # 636d  OR 2e29
MODEL_NAME="${INPUT_DATASET%_eval_*}"

# Build output-dataset path
SUFFIX="_eval_${HASH}"
MODEL_BASENAME="${MODEL_NAME##*/}"
OUTPUT_DATASET="mlfoundations-dev/${MODEL_BASENAME}${SUFFIX}"

# 2. TASK SET
STANDARD_EVAL_TASKS=(AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces)
FULL_EVAL_TASKS=(AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces,AIME25,HLE,LiveCodeBenchv5)
AIME24=(AIME24)

if   [[ "${HASH}" == "636d" ]]; then TASKS=("${STANDARD_EVAL_TASKS[@]}")
elif [[ "${HASH}" == "2e29" ]]; then TASKS=("${FULL_EVAL_TASKS[@]}")
else echo "❌  Unknown evaluation hash '${HASH}'"; exit 1;
fi
TASKS_STR=$(IFS=, ; echo "${TASKS[*]}")

# 3. LOAD MODULES & ACTIVATE ENV ──────────────────────────────────────────────
set -e

source /work/10159/rmarten/vista/dcft/dcft_private/hpc/dotenv/tacc.env
$EVALCHEMY_ACTIVATE_ENV

echo "┌──────────────────────────────────────────────"
echo "│  MODEL      : ${MODEL_NAME}"
echo "│  TASKS      : ${TASKS_STR}"
echo "│  OUT DATASET: ${OUTPUT_DATASET}"
echo "└──────────────────────────────────────────────"

srun --nodes=1 --ntasks=1 python -m eval.eval \
  --model precomputed_hf \
  --model_args "repo_id=${OUTPUT_DATASET},model=${MODEL_NAME}" \
  --tasks "${TASKS_STR}" \
  --output_path logs \
  --use_database
