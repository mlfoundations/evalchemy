#!/usr/bin/env python3
"""
Generate individual evaluation scripts for each model-task combination.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration for generated SLURM scripts
# ---------------------------------------------------------------------------
# Number of nodes to request in SLURM.  Use 1 for single-node jobs.
NUM_NODES = 1

# Number of GPUs per node.
GPUS_PER_NODE = 4

EXPERIMENT_PATH = "/e/project1/jureap59/ali/post-training/outputs/sft-olmo-3-1025-7b-nemotron_pt_v2-20260221-162923/inference_checkpoints"


# Define the models (only uncommented/active models are used to generate scripts)
MODELS = [
    # "ali-elganzory/SmolLM2-1.7B-WithChatTemplate",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "Qwen/Qwen2.5-1.5B",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen3-1.7B-Base",
    # "Qwen/Qwen3-1.7B",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-WithChatTemplate",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate",
    # "ali-elganzory/open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384-WithChatTemplate",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-SFT-Tulu3",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-DPO-Tulu3",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-DPO-Tulu3",
    # "ali-elganzory/open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384-SFT-Tulu3",
    # "ali-elganzory/open-sci-ref-v0.01-1.7b-nemotron-hq-300B-16384-DPO-Tulu3",
    # "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
    # "ali-elganzory/Qwen2.5-1.5B-SFT-Tulu3-decontaminated",
    # "ali-elganzory/Qwen3-1.7B-Base-SFT-Tulu3-decontaminated",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct-16k",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
    ############################################################
    ############################################################
    # "ali-elganzory/SmolLM2-1.7B-DPO-Tulu3-decontaminated",
    # "ali-elganzory/Qwen2.5-1.5B-DPO-Tulu3-decontaminated",
    # "ali-elganzory/Qwen3-1.7B-Base-DPO-Tulu3-decontaminated",
    # ###
    # "ali-elganzory/1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-100BT-DPO-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-DPO-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-DPO-Tulu3-decontaminated",
    # ###
    # "ontocord/SmolLM2-1.7B-16k",
    # "ali-elganzory/SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated",
    # "ali-elganzory/SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated",
    # ###
    # "ali-elganzory/1.7b-MixtureVitae-100BT",
    # "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT",
    # "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT",
    # ###
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated",
    ############################################################
    # "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate",
    # "ali-elganzory/1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated",
    # "ali-elganzory/1.7b-Comma0.1-300BT-DPO-Tulu3-decontaminated",
    # ###
    # "ali-elganzory/ablation-model-fineweb-edu-WithChatTemplate",
    # "ali-elganzory/ablation-model-fineweb-edu-SFT-Tulu3-decontaminated",
    # "ali-elganzory/ablation-model-fineweb-edu-DPO-Tulu3-decontaminated",
    ############################################################
    *[
        f"{EXPERIMENT_PATH}/{model}"
        for model in os.listdir(EXPERIMENT_PATH)
        if os.path.isdir(f"{EXPERIMENT_PATH}/{model}")
    ]
]

# Define the tasks
TASKS = [
    # "IFEval",
    # "HumanEval",
    # "MBPP",
    # "AIME24",
    # "AIME25",
    # "AMC23",
    # "MATH500",
    # "LiveCodeBench",
    # "GPQADiamond",
    # "JEEBench",
    ####################
    "alpaca_eval",
]


def generate_model_names_section(active_model: str) -> str:
    """Generate the MODEL_NAMES array with only the active model uncommented."""
    # Only output the active model (no commented-out lines for excluded models)
    lines = ["MODEL_NAMES=("]
    lines.append(f'    "{active_model}"')
    lines.append(")")
    return "\n".join(lines)


def generate_tasks_section(active_task: str) -> str:
    """Generate the TASKS array with only the active task uncommented."""
    lines = ["TASKS=("]
    lines.append(f'    "{active_task}"')
    lines.append(")")
    return "\n".join(lines)


def generate_script(model: str, task: str) -> str:
    """Generate the full script content for a model-task combination."""

    model_names_section = generate_model_names_section(model)
    tasks_section = generate_tasks_section(task)

    # Multi-node specific logic: only included in generated scripts when
    # NUM_NODES > 1 to keep single-node scripts minimal.
    if NUM_NODES > 1:
        multi_node_logic = f"""
# 1. Set master
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR="${{MASTER_ADDR}}"
export MASTER_IP=$(nslookup $MASTER_ADDR | awk '/^Address: / {{ print $2 }}' | tail -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 2000))

if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: Could not find MASTER_ADDR. hostname -I returned empty."
    exit 1
fi

# 2. High-performance tuning for NCCL on multi-rail IB
export NCCL_IB_HCA=mlx5
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=120
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1


# 3. Calculate World Size
export NUM_NODES=$SLURM_NNODES
export GPUS_PER_NODE={GPUS_PER_NODE}
export WORLD_SIZE=$(($NUM_NODES * $GPUS_PER_NODE))

echo "Master Node: $MASTER_ADDR"
echo "Master IP: $MASTER_IP"
echo "Master Port: $MASTER_PORT"
echo "Network Interface: $NCCL_SOCKET_IFNAME"
"""
        # We use srun to start one task per node, and let Accelerate handle
        # spawning per-GPU processes on each node. SLURM provides SLURM_PROCID
        # which we pass as machine_rank.
        accelerate_launch = """srun --nodes "$NUM_NODES" --ntasks "$NUM_NODES" --ntasks-per-node 1 \\
            accelerate launch --num_machines $NUM_NODES \\
            --num_processes $WORLD_SIZE \\
            --main_process_ip $MASTER_IP \\
            --main_process_port $MASTER_PORT \\
            --machine_rank $SLURM_PROCID \\
            --same_network \\
            --max_restarts 0 \\
            --role $(hostname -s): \\
            --multi_gpu -m eval.eval \\"""  # noqa: E501
    else:
        multi_node_logic = ""
        accelerate_launch = """accelerate launch --num-processes "$GPUS_PER_NODE" --num-machines 1 \\
            --multi-gpu -m eval.eval \\"""  # noqa: E501

    # Create a safe name for the job
    model_short = model.split("/")[-1]
    job_name = f"eval_{model_short}_{task}"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/mv_exp/{model_short}/{task}_%j.%x.%N.out
#SBATCH --error=slurm_logs/mv_exp/{model_short}/{task}_%j.%x.%N.err
#SBATCH --time=00-12:00:00
#SBATCH --partition=booster
#SBATCH --account=jureap59
#SBATCH --nodes={NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:{GPUS_PER_NODE}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com

# Resource configuration (can be tweaked per-script if needed)
NUM_NODES={NUM_NODES}
GPUS_PER_NODE={GPUS_PER_NODE}

{multi_node_logic}

export WORK_DIR=/e/project1/jureap59/ali
export EVALCHEMY_DIR=$WORK_DIR/evalchemy
export TMPDIR=$WORK_DIR/.tmp
export TMP=$TMPDIR

module load CUDA/13

# Force Transformers and Hub into offline mode
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# If you are using Hugging Face Datasets as well
export HF_DATASETS_OFFLINE=1

export WANDB_MODE="offline"

# 1. Evaluation arguments
{model_names_section}
{tasks_section}

# 2. Go to evalchemy directory
cd $EVALCHEMY_DIR

# 3. Activate the environment
source .venv/bin/activate

# 4. Run evaluation
for MODEL in "${{MODEL_NAMES[@]}}"; do
    for TASK in "${{TASKS[@]}}"; do
        echo "==========================================================="
        echo "Starting evaluation"
        echo "Model: $MODEL"
        echo "Task: $TASK"
        echo "==========================================================="
        {accelerate_launch}
            --model hf \\
            --tasks "$TASK" \\
            --model_args "trust_remote_code=True,pretrained=$MODEL" \\
            --batch_size auto \\
            --result_dir outputs/
    done
done
"""
    return script


def get_safe_filename(model: str, task: str) -> str:
    """Generate a safe filename for the script."""
    # Extract the model name after the slash and make it filesystem-safe
    model_name = model.split("/")[-1]
    # Replace any problematic characters
    model_name = model_name.replace("/", "_").replace(" ", "_")
    return f"eval_{model_name}_{task}.sh"


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Create the eval_scripts folder
    eval_scripts_dir = script_dir / "eval_scripts"
    eval_scripts_dir.mkdir(exist_ok=True)

    print(f"Creating evaluation scripts in: {eval_scripts_dir}")
    print(f"Models: {len(MODELS)}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Total scripts to generate: {len(MODELS) * len(TASKS)}")
    print()

    script_count = 0

    for model in MODELS:
        for task in TASKS:
            filename = get_safe_filename(model, task)
            filepath = eval_scripts_dir / filename

            script_content = generate_script(model, task)

            with open(filepath, "w") as f:
                f.write(script_content)

            # Make the script executable
            os.chmod(filepath, 0o755)

            script_count += 1
            n_scripts = len(MODELS) * len(TASKS)
            num_digits = len(str(n_scripts))
            print(f"[{script_count:0{num_digits}d}/{n_scripts}] Generated: {filename}")

    print()
    print(f"Successfully generated {script_count} scripts in {eval_scripts_dir}")


if __name__ == "__main__":
    main()
