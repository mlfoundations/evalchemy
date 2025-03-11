# Capella HPC Cluster Setup

This document details the setup process for running distributed evaluations on the Capella HPC cluster.

## Important Notes

- **Internet access**: Unlike Leonardo, Capella compute nodes have internet access
- **Shared workspace**: Setup uses a shared workspace for collaboration and efficient resource usage
- **Standard conda**: Uses standard Miniconda instead of Mamba

## Environment Setup

Follow these steps to set up the environment on Capella:

```bash
# Allocate a workspace
# Guide: https://doc.zih.tu-dresden.de/quickstart/getting_started/?h=workspaces#allocate-a-workspace
ws_allocate -F horse -r 7 -m ryanmarten2000@gmail.com -n DCFT_Shared -d 100
echo "export DCFT=/data/horse/ws/ryma833h-DCFT_Shared" >> ~/.bashrc

# Set up shared access for project members
# Check your group membership
groups # should show p_finetuning
ls -ld $DCFT # Should show p_finetuning ownership

# Set appropriate permissions for collaboration
# Owner and group get full access, others have no access
chmod -R u+rwX,g+rwX,o-rwx $DCFT
# Set default ACLs for new files to maintain these permissions
setfacl -R -d -m u::rwX,g::rwX,o::- $DCFT

# Set up conda in the shared workspace
mkdir -p $DCFT/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $DCFT/miniconda3/miniconda.sh
bash $DCFT/miniconda3/miniconda.sh -b -u -p $WORK/miniconda3
rm $DCFT/miniconda3/miniconda.sh
source $DCFT/miniconda3/bin/activate
conda init  # Adds conda initialization to ~/.bashrc

# Clone repo and create conda environment
git clone git@github.com:mlfoundations/evalchemy.git $DCFT/evalchemy
cd $DCFT/evalchemy
conda create -y --name evalchemy python=3.10
conda activate evalchemy
pip install -e .
pip install -e eval/chat_benchmarks/alpaca_eval

# Setup shared database access
cat << 'EOF' >> $EVALCHEMY/.env
export DB_PASSWORD=XXX
export DB_HOST=XXX
export DB_PORT=XXX
export DB_NAME=XXX
export DB_USER=XXX
EOF

# Create shared HuggingFace cache
mkdir -p $DCFT/huggingface/hub
echo "export HF_HUB_CACHE=$DCFT/huggingface/hub" >> ~/.bashrc
```

## Testing the Setup

Before running full distributed evaluations, test the setup:

```bash
# Test the basic installation with a simple task
OPENAI_API_KEY=NONE python -m eval.eval --model upload_to_hf --tasks AIME25 --model_args repo_id=mlfoundations-dev/AIME25_evalchemy

# Test the launcher with minimal configuration
python eval/distributed/launch.py --model_name open-thoughts/OpenThinker-7B --tasks AIME24 --num_shards 1 --watchdog
```

## Running Distributed Evaluations

To run a distributed evaluation on Capella:

```bash
# Activate the environment
source /data/horse/ws/ryma833h-DCFT_Shared/miniconda3/bin/activate
conda activate evalchemy
cd /data/horse/ws/ryma833h-DCFT_Shared/evalchemy

# Launch the distributed evaluation
python eval/distributed/launch.py --model_name open-thoughts/OpenThinker-7B --tasks LiveCodeBench,AIME24,AIME25,AMC23,GPQADiamond,MATH500 --num_shards 8 --max-job-duration 2 --watchdog
```

The distributed launcher will:
1. Detect the Capella environment
2. Use the appropriate HuggingFace cache location
3. Use the correct sbatch script for Capella
4. Monitor the job progress (with `--watchdog`)
5. Upload results when complete

## Monitoring Jobs

You can monitor your jobs using standard SLURM commands:

```bash
# Check job status
squeue -u $USER

# Get detailed job information
sacct -j <job_id> -X --format=JobID,JobName,State,Elapsed

# Cancel a job if needed
scancel <job_id>
```
