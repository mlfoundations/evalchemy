#!/bin/bash -x
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time={time_limit}
#SBATCH --job-name={job_name}
#SBATCH --exclude=lrdn[1606,2776,2425,2808,3064,3064,1953,2414,1506,1718,1779,2828,2354,3279,1370,2595,2751,2921,2368,2976,2733,2277,3136,2013,2952,1427,2682,2349,1655,1390,3151,3130,2002,2654,2101,2358,1597,2585,2900,2687,3165,3031,2798,2530,2344,1384,1420,1474,1509,1520,1556,1607,1647,1810,1927,2000,2028,2056,2120,2136,2371,2384,2444,2465,2479,2563,2598,2652,2716,2731,2746,2755,2772,2775,2792,2794,2917,2926,2927,3110,3221,3395,0666]
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=dcft-slurm-notifs-aaaap7wt363mcsgryaejj2o6dm@dogs-and-ml.slack.com

# EXIT ON FAILURE
set -e

# MODULES
module load cuda/12.4 nccl/12.4

# ENVIRONMENT VARIABLES - EVALCHEMY, HF_HUB_CACHE, and EVALCHEMY_ACTIVATE_ENV
# source /work/10159/rmarten/vista/dcft/dcft_private/hpc/dotenv/tacc.env
source /leonardo_work/EUHPC_E03_068/DCFT_shared/dcft_private/hpc/dotenv/leonardo.env
source /leonardo_work/EUHPC_E03_068/DCFT_shared/mamba/bin/activate /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy

# CONDA
$EVALCHEMY_ACTIVATE_ENV

# DOWNLOAD MODEL AND DATASET
MODEL_NAME={model_name}
INPUT_DATASET={input_dataset}
OUTPUT_DATASET={output_dataset}
srun --nodes=1 huggingface-cli download $MODEL_NAME --cache-dir $HF_HUB_CACHE
srun --nodes=1 huggingface-cli download $INPUT_DATASET --cache-dir $HF_HUB_CACHE --repo-type dataset

# RUN SHARDED INFERENCE
srun --output={logs_dir}/%x_%j_%n.out bash -c 'echo -e "GLOBAL_SIZE: ${SLURM_JOB_NUM_NODES}\nRANK: ${SLURM_NODEID}\nMODEL: '$MODEL_NAME'\nINPUT_DATASET: '$INPUT_DATASET'\nOUTPUT_DATASET: '$OUTPUT_DATASET'"'
srun --output={logs_dir}/%x_%j_%n.out bash -c 'python $EVALCHEMY/eval/distributed/process_shard.py --global_size ${SLURM_JOB_NUM_NODES} --rank ${SLURM_NODEID} --input_dataset '${INPUT_DATASET}' --model_name '${MODEL_NAME}' --output_dataset '${OUTPUT_DATASET}' --upload'

# COMPUTE SCORES
srun --nodes=1 python -m eval.eval --model precomputed_hf --model_args "repo_id={output_dataset}",model="{model_name}" --tasks {tasks_str} --output_path logs --use_database
