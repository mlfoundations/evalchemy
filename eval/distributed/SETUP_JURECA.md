
# Initial env setup
Stored already in `jureca.env`
```
export DCFT=/p/project1/laionize/dcft
export DCFT_DATA=/p/data1/mmlaion/dcft
export DCFT_GROUP=laionize
export HF_HUB_CACHE=$DCFT_DATA/hub
export DCFT_CONDA=$DCFT_DATA/mamba
export EVALCHEMY=$DCFT/evalchemy
export EVALCHEMY_ENV=$DCFT_DATA/evalchemy_env
export EVALCHEMY_ACTIVATE_ENV="source $DCFT_CONDA/bin/activate $EVALCHEMY_ENV"
```
And in `common.sh`
```
if [[ "$PRIMARY_GROUP_SET" != "true" ]]; then
  export PRIMARY_GROUP_SET=true
  exec newgrp $DCFT_GROUP
fi
umask 007
```
And previously setup
```
cd /p/project1/laionize
mkdir dcft
chmod g+s dcft
chmod g+rwX dcft
newgrp laionize
umask 007

git clone https://USER:TOKEN@github.com/mlfoundations/evalchemy.git
cd evalchemy
git remote set-url origin https://USER:TOKEN@github.com/mlfoundations/evalchemy.git
cd ..

git clone https://USER:TOKEN@github.com/mlfoundations/dcft_private.git
cd dcft_private 
git remote set-url origin https://USER:TOKEN@github.com/mlfoundations/dcft_private.git

cd /p/data1/mmlaion/dcft
chmod g+s dcft
chmod g+rwX dcft
mkdir hub
chgrp -R mmlaion hub
chmod g+s hub
chmod g+rwX hub
mkdir checkpoints
chgrp -R mmlaion checkpoints
chmod g+s checkpoints
chmod g+rwX checkpoints
```

Setup commands
```
cd $DCFT_DATA
mkdir -p evalchemy_results

# Install Mamba (following Jenia's guide: https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Creating-env)
SHELL_NAME=bash
VERSION=23.3.1-0

# NOTE: download the exact python version and --clone off base
curl -L -O "https://github.com/conda-forge/miniforge/releases/download/${VERSION}/Mambaforge-${VERSION}-$(uname)-$(uname -m).sh" 
chmod +x Mambaforge-${VERSION}-$(uname)-$(uname -m).sh
./Mambaforge-${VERSION}-$(uname)-$(uname -m).sh -b -p $DCFT_CONDA
chgrp -R mmlaion mamba
rm ./Mambaforge-${VERSION}-$(uname)-$(uname -m).sh
eval "$(${DCFT_CONDA}/bin/conda shell.${SHELL_NAME} hook)"
${DCFT_CONDA}/bin/mamba create -y --prefix ${EVALCHEMY_ENV} --clone base
source ${DCFT_CONDA}/bin/activate ${EVALCHEMY_ENV}


# Fix path resolution issue in the installation
cd  $EVALCHEMY
sed -i 's|"fschat @ file:eval/chat_benchmarks/MTBench"|"fschat @ file:///p/project1/laionize/dcft/evalchemy/eval/chat_benchmarks/MTBench"|g' /p/project1/laionize/dcft/evalchemy/pyproject.toml
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
```

# Test the sbatch script
```
sbatch eval/distributed/process_shards_jureca.sbatch
# Clean up logs when done
rm *.out
```

# Test the launch
```
python eval/distributed/launch.py --model_name open-thoughts/OpenThinker-7B --tasks AIME24 --num_shards 16 
```
NOTE: No node sharing so use the simple launch instead
```
python eval/distributed/launch_simple.py --model_name open-thoughts/OpenThinker-7B --tasks AIME24 --num_shards 16
```

## Test the upload
```
huggingface-cli upload mlfoundations-dev/OpenThinker-7B_eval_2870 $EVALCHEMY_RESULTS_DIR/OpenThinker-7B_eval_2870 --repo-type=dataset --commit-message="upload inferences" 
```
