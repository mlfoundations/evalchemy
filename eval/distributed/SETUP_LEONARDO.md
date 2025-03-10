# Leonardo HPC Cluster Setup

This document details the setup process for running distributed evaluations on the Leonardo HPC cluster.

## Important Notes

- **No internet access**: Leonardo compute nodes do not have internet access, so all dependencies must be downloaded in advance
- **Shared environment**: The setup uses a shared workspace for better collaboration
- **Offline mode**: The code is configured to run in offline mode on compute nodes

## Environment Setup

Follow these steps to set up the environment on Leonardo:

```bash
# Set up environment variables in your .bashrc for easier access
cat << 'EOF' >> ~/.bashrc
DCFT=$WORK/DCFT_shared
EVALCHEMY=$DCFT/evalchemy/
DCFT_MAMBA=$DCFT/mamba
EVALCHEMY_GPU_ENV=$EVALCHEMY/env/cu121-evalchemy
EVALCHEMY_CPU_ENV=$EVALCHEMY/env/cpu-evalchemy
EOF

# Install Mamba (following Jenia's guide: https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Creating-env)
SHELL_NAME=bash
VERSION=23.3.1-0

# Create shared database access
cat << 'EOF' >> $EVALCHEMY/.env
export DB_PASSWORD=XXX
export DB_HOST=XXX
export DB_PORT=XXX
export DB_NAME=XXX
export DB_USER=XXX
EOF

# Create the shared conda environment
cd $EVALCHEMY
module purge
module load cuda/12.1
module load gcc/12.2.0
module load nccl

# NOTE: download the exact python version and --clone off base
# This avoids using conda servers (getting 429 rate limit from Leonardo IP address)
curl -L -O "https://github.com/conda-forge/miniforge/releases/download/${VERSION}/Mambaforge-${VERSION}-$(uname)-$(uname -m).sh" 
chmod +x Mambaforge-${VERSION}-$(uname)-$(uname -m).sh
./Mambaforge-${VERSION}-$(uname)-$(uname -m).sh -b -p $DCFT_MAMBA
rm ./Mambaforge-${VERSION}-$(uname)-$(uname -m).sh
eval "$(${DCFT_MAMBA}/bin/conda shell.${SHELL_NAME} hook)"
${DCFT_MAMBA}/bin/mamba create -y --prefix ${EVALCHEMY_GPU_ENV} --clone base
source ${DCFT_MAMBA}/bin/activate ${EVALCHEMY_GPU_ENV}

# Fix path resolution issue in the installation
sed -i 's|"fschat @ file:eval/chat_benchmarks/MTBench"|"fschat @ file:///leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/eval/chat_benchmarks/MTBench"|g' /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/pyproject.toml
pip install -e .
pip install -e eval/chat_benchmarks/alpaca_eval
git reset --hard HEAD

# Create another shared environment for CPU only
${DCFT_MAMBA}/bin/mamba create -y --prefix ${EVALCHEMY_CPU_ENV} --clone ${EVALCHEMY_GPU_ENV}
source ${DCFT_MAMBA}/bin/activate ${EVALCHEMY_CPU_ENV}
pip uninstall -y torch torchvision torchaudio && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Test CPU environment
export HF_HUB_CACHE="$DCFT/hub"
OPENAI_API_KEY=NONE python -m eval.eval --model upload_to_hf --tasks AIME25 --model_args repo_id=mlfoundations-dev/AIME25_evalchemy

# Install the right torch for GPU env
source ${DCFT_MAMBA}/bin/activate ${EVALCHEMY_GPU_ENV}
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Testing the Setup

Before running full distributed evaluations, test the setup with:

```bash
# Download necessary datasets and models on the login node
huggingface-cli download mlfoundations-dev/evalset_2870 --repo-type dataset
huggingface-cli download open-thoughts/OpenThinker-7B

# Request an interactive node for testing
salloc --nodes=1 --ntasks-per-node=1 --account=EUHPC_E03_068 --partition=boost_usr_prod --qos=boost_qos_dbg --gres=gpu:1

# Verify GPU is available
srun bash -c 'nvidia-smi'

# Test the inference pipeline manually
# Run through commands similar to those in eval/distributed/process_shards_leonardo.sbatch
export GLOBAL_SIZE=16
export RANK=0
# Run shard processing script...

# Test the sbatch script
sbatch eval/distributed/process_shards_leonardo.sbatch
# Clean up logs when done
rm *.out
```

## Benchmarking Shard Performance

Before running full distributed evaluations, it's recommended to benchmark performance with different shard counts to determine the optimal configuration for your workload. The benchmark helps identify the best trade-off between wall-clock time and total GPU hours.

```bash
# Activate the CPU environment 
source /leonardo_work/EUHPC_E03_068/DCFT_shared/mamba/bin/activate /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy

# Run benchmark with different shard counts
# Replace N with the number of shards to test (e.g., 2, 4, 8, 16, 32, 64, 128)
SHARDS=N && cd $EVALCHEMY && source /leonardo_work/EUHPC_E03_068/DCFT_shared/mamba/bin/activate /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy && python eval/distributed/launch.py --model_name open-thoughts/OpenThinker-7B --tasks LiveCodeBench,AIME24,AIME25,AMC23,GPQADiamond,MATH500 --num_shards $SHARDS --watchdog 
```

Our benchmarks on Leonardo with a standard reasoning evaluation workload show:

<img src="./benchmarking_leonardo.png" alt="Benchmarking Example" width="50%"/>

Key insights from benchmarking:
- 8 shards provides a good balance between execution time and GPU efficiency
- More than 64 shards shows diminishing returns on time reduction
- Total GPU hours increases significantly with more than 16 shards

## Running Distributed Evaluations

To run a distributed evaluation on Leonardo:

```bash
# Activate the CPU environment (the launcher will run on the login node)
source /leonardo_work/EUHPC_E03_068/DCFT_shared/mamba/bin/activate /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy

# Launch the distributed evaluation
python eval/distributed/launch.py --model_name open-thoughts/OpenThinker-7B --tasks LiveCodeBench,AIME24,AIME25,AMC23,GPQADiamond,MATH500 --num_shards 8 --max-job-duration 2 --watchdog
```

The distributed launcher will:
1. Detect the Leonardo environment
2. Use the appropriate HuggingFace cache location
3. Use the correct sbatch script for Leonardo
4. Monitor the job progress (with `--watchdog`)
5. Upload results when complete