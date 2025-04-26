
# Initial env setup
```
# Install Mamba (following Jenia's guide: https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Creating-env)
SHELL_NAME=bash
VERSION=23.3.1-0

# NOTE: download the exact python version and --clone off base
curl -L -O "https://github.com/conda-forge/miniforge/releases/download/${VERSION}/Mambaforge-${VERSION}-$(uname)-$(uname -m).sh" 
chmod +x Mambaforge-${VERSION}-$(uname)-$(uname -m).sh
./Mambaforge-${VERSION}-$(uname)-$(uname -m).sh -b -p $DCFT_MAMBA
rm ./Mambaforge-${VERSION}-$(uname)-$(uname -m).sh
eval "$(${DCFT_CONDA}/bin/conda shell.${SHELL_NAME} hook)"
${DCFT_CONDA}/bin/mamba create -y --prefix ${EVALCHEMY_ENV} --clone base
source ${DCFT_CONDA}/bin/activate ${EVALCHEMY_ENV}

# Fix path resolution issue in the installation
sed -i 's|"fschat @ file:eval/chat_benchmarks/MTBench"|"fschat @ file:///leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/eval/chat_benchmarks/MTBench"|g' /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/pyproject.toml
pip install -e .
pip install -e eval/chat_benchmarks/alpaca_eval
git reset --hard HEAD

python -m eval.eval --model upload_to_hf --tasks AIME24 --model_args repo_id=mlfoundations-dev/evalset_2870
```

# Test processing shards
```
# Download necessary datasets and models on the login node
# Note that HF_HUB_CACHE needs to be set as it is above
huggingface-cli download mlfoundations-dev/evalset_2870 --repo-type dataset
huggingface-cli download open-thoughts/OpenThinker-7B

# Request an interactive node for testing
salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=12 -p dc-hwai -A westai0007

# Verify GPU is available
srun bash -c 'nvidia-smi'

# Test the inference pipeline manually
# Run through commands similar to those in eval/distributed/process_shards_jureca.sbatch
mkdir -p results
export GLOBAL_SIZE=32
export RANK=0
export MODEL_NAME_SHORT=$(echo "$MODEL_NAME" | sed -n 's/.*models--[^-]*--\([^\/]*\).*/\1/p')
export INPUT_DATASET="$HF_HUB_CACHE/datasets--mlfoundations-dev--evalset_2870"
export OUTPUT_DATASET="$EVALCHEMY/results/${MODEL_NAME_SHORT}_${INPUT_DATASET##*--}"
srun echo -e "GLOBAL_SIZE: ${GLOBAL_SIZE}\nRANK: ${RANK}\nMODEL: ${MODEL_NAME}\nINPUT_DATASET: ${INPUT_DATASET}\nOUTPUT_DATASET: ${OUTPUT_DATASET}"

# Test the sbatch script
sbatch eval/distributed/process_shards_leonardo.sbatch
# Clean up logs when done
rm *.out
```
