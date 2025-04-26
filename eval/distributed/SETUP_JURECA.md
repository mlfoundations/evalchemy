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
