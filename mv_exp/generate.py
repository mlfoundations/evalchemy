#!/usr/bin/env python3
"""
Generate individual evaluation scripts for each model-task combination.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration for generated SLURM scripts
# ---------------------------------------------------------------------------
# Number of nodes to request in SLURM.  Use 1 for single-node jobs.
NUM_NODES = 1

# Number of GPUs per node.
GPUS_PER_NODE = 1

# Optional: set on multi-node if Jupiter/cluster docs require a specific host interface
# for NCCL or for Gloo/TCP rendezvous (the latter runs before NCCL collectives).
NCCL_SOCKET_IFNAME = ""
GLOO_SOCKET_IFNAME = ""

# When True, shorten distributed timeouts and enable verbose dist logs (debugging only).
DEBUG_SHORT_DIST_TIMEOUT = False

EXPERIMENT_PATH = "/e/project1/jureap59/ali/post-training/outputs/sft-olmo-3-1025-7b-nemotron_pt_v2-20260221-162923/inference_checkpoints"

BATCH_SIZE_CACHE_FILENAME = "batch_size_cache.json"
DISCOVERY_SCRIPTS_DIRNAME = "batch_size_discovery_scripts"


# Define the models (only uncommented/active models are used to generate scripts)
MODELS = [
    # ---------------------------------------------------------
    # Base Models
    # ---------------------------------------------------------
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16384-rope_theta-1M-long_sft_16k",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k",
    "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate",
    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k",
    "ali-elganzory/SmolLM2-1.7B-WithChatTemplate",
    "ali-elganzory/SmolLM2-1.7B-16k",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-1.7B-Base",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k",
    "ali-elganzory/1.7b-MixtureVitae-100BT",
    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-WithChatTemplate",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate",
    "ali-elganzory/Baguettotron",
    "ali-elganzory/Baguettotron-longsft_16k",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k",
    # ---------------------------------------------------------
    # SFT Models (100% Finished)
    # ---------------------------------------------------------
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/Qwen2.5-1.5B-SFT-Tulu3-decontaminated",
    "ali-elganzory/Qwen3-1.7B-Base-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-SFT-Tulu3",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3",
    "ali-elganzory/Baguettotron-SFT-Tulu3-decontaminated",
    "ali-elganzory/Baguettotron-longsft_16k-SFT-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-SFT-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    # ---------------------------------------------------------
    # DPO Models (100% Finished)
    # ---------------------------------------------------------
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-DPO-Tulu3-decontaminated",
    "ali-elganzory/SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/Qwen2.5-1.5B-DPO-Tulu3-decontaminated",
    "ali-elganzory/Qwen3-1.7B-Base-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-DPO-Tulu3",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-DPO-Tulu3",
    "ali-elganzory/Baguettotron-DPO-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-DPO-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/Baguettotron-longsft_16k-DPO-Tulu3-decontaminated",
    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    # ---------------------------------------------------------
    # Merged Models (100% Finished)
    # ---------------------------------------------------------
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-SFT-Tulu3-decontaminated",
    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-DPO-Tulu3-decontaminated",
    # ---------------------------------------------------------
    # OT3 Models (100% Finished)
    # ---------------------------------------------------------
    "open-sci/sft_ot30k_1.7b-MixtureVitae-300BT-v1-decontaminated-16k_base",
]

# Define the tasks
TASKS = [
    # "IFEval",
    # "HumanEval",
    # "MBPP",
    # "AIME24",
    # "AIME25",
    # "AMC23",
    "gsm8k",
    # "MATH500",
    # "LiveCodeBench",
    # "GPQADiamond",
    # "JEEBench",
]


def _multi_node_socket_exports_bash() -> str:
    """Optional export lines for NCCL / Gloo socket interface (empty constants = no lines)."""
    lines: list[str] = []
    if NCCL_SOCKET_IFNAME.strip():
        lines.append(f'export NCCL_SOCKET_IFNAME="{NCCL_SOCKET_IFNAME.strip()}"')
    if GLOO_SOCKET_IFNAME.strip():
        lines.append(f'export GLOO_SOCKET_IFNAME="{GLOO_SOCKET_IFNAME.strip()}"')
    return ("\n".join(lines) + "\n") if lines else ""


def _multi_node_debug_timeout_bash() -> str:
    if DEBUG_SHORT_DIST_TIMEOUT:
        return """
# Shorter distributed timeouts for faster feedback while debugging (not for production).
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=0:03:0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
"""
    return """
# Debug: uncomment for faster rendezvous failure or richer dist logs (verify names for your PyTorch build).
# export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=0:03:0
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# torchrun also exposes --rdzv_timeout (seconds) if you bypass Accelerate.
"""


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


def get_batch_size_cache_path(script_dir: Path) -> Path:
    return script_dir / BATCH_SIZE_CACHE_FILENAME


def load_batch_size_cache(cache_path: Path) -> dict[str, dict[str, int]]:
    if not cache_path.exists():
        return {}

    try:
        raw_text = cache_path.read_text().strip()
        raw_cache = {} if not raw_text else json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in batch size cache {cache_path}: {e}") from e

    cache: dict[str, dict[str, int]] = {}
    for model, task_map in raw_cache.items():
        if not isinstance(model, str) or not isinstance(task_map, dict):
            continue

        validated_task_map: dict[str, int] = {}
        for task, batch_size in task_map.items():
            if isinstance(task, str) and isinstance(batch_size, int) and batch_size > 0:
                validated_task_map[task] = batch_size

        if validated_task_map:
            cache[model] = validated_task_map

    return cache


def resolve_cached_batch_size(
    batch_size_cache: dict[str, dict[str, int]], model: str, task: str
) -> int | None:
    return batch_size_cache.get(model, {}).get(task)


def generate_discovery_script(model: str, task: str) -> str:
    model_names_section = generate_model_names_section(model)
    tasks_section = generate_tasks_section(task)

    model_short = model.split("/")[-1]
    job_name = f"discover_bs_{model_short}_{task}"

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/mv_exp/{model_short}/{task}_discover_bs_%j.%x.%N.out
#SBATCH --error=slurm_logs/mv_exp/{model_short}/{task}_discover_bs_%j.%x.%N.err
#SBATCH --time=00-01:00:00
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com

export WORK_DIR=/e/project1/reformo/ali
export EVALCHEMY_DIR=$WORK_DIR/evalchemy
export TMPDIR=$WORK_DIR/.tmp
export TMP=$TMPDIR

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="offline"

{model_names_section}
{tasks_section}

cd $EVALCHEMY_DIR

module load Stages/2025
module load CUDA/12
source .venv/bin/activate

export SRUN_CPUS_PER_TASK=18

for MODEL in "${{MODEL_NAMES[@]}}"; do
    for TASK in "${{TASKS[@]}}"; do
        echo "==========================================================="
        echo "Starting batch-size discovery"
        echo "Model: $MODEL"
        echo "Task: $TASK"
        echo "==========================================================="
        srun --ntasks=1 --export=ALL --wait=60 --kill-on-bad-exit=1 python mv_exp/discover_batch_size.py \\
            --model "$MODEL" \\
            --task "$TASK" \\
            --cache-file "$EVALCHEMY_DIR/mv_exp/{BATCH_SIZE_CACHE_FILENAME}" \\
            --output-path logs
    done
done
"""


def generate_script(model: str, task: str, cached_batch_size: int | str | None) -> str:
    """Generate the full script content for a model-task combination."""

    model_names_section = generate_model_names_section(model)
    tasks_section = generate_tasks_section(task)

    socket_exports = _multi_node_socket_exports_bash()
    debug_timeout = _multi_node_debug_timeout_bash()

    # Multi-node specific logic: only included in generated scripts when
    # NUM_NODES > 1 to keep single-node scripts minimal.
    if NUM_NODES > 1:
        multi_node_logic = f"""
# 1. Set master (batch script runs on first allocated node)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR="${{MASTER_ADDR}}"
# Prefer IPv4 from NSS (avoids IPv6 / ambiguous nslookup Address lines for Gloo TCP rendezvous).
MASTER_IP=$(getent ahostsv4 "$MASTER_ADDR" | awk '/^[0-9]/ {{print $1; exit}}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(nslookup "$MASTER_ADDR" | awk '/^Address: / {{ print $2 }}' | tail -n 1)
fi
export MASTER_IP
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 2000))

if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: Could not find MASTER_ADDR."
    exit 1
fi

# 2. High-performance tuning for NCCL on multi-rail IB
export NCCL_IB_HCA=mlx5
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=120
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# Gloo/TCP (elastic rendezvous) may need GLOO_SOCKET_IFNAME; NCCL_* affects GPU collectives after rendezvous.
{socket_exports}
# 3. Calculate world size
export NUM_NODES=$SLURM_NNODES
export GPUS_PER_NODE={GPUS_PER_NODE}
export WORLD_SIZE=$(($NUM_NODES * $GPUS_PER_NODE))

echo "Master Node: $MASTER_ADDR"
echo "Master IP: $MASTER_IP"
echo "Master Port: $MASTER_PORT"
echo "NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME:-}}  GLOO_SOCKET_IFNAME=${{GLOO_SOCKET_IFNAME:-}}"

# Rendezvous: if c10d logs still show hostname, ensure workers reach $MASTER_IP:$MASTER_PORT; set ifnames per site docs.
{debug_timeout}"""
        # Match alignment-handbook jupiter_sft.slurm.j2: LAUNCHER + CMD, then
        # srun ... bash -c so each node runs one shell that expands $SLURM_PROCID locally.
        if cached_batch_size is None:
            cached_batch_size = "auto"

        run_eval_inner = rf"""        export LAUNCHER="accelerate launch \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --same_network \
    --max_restarts 0 \
    --role \$(hostname -s): \
    --multi_gpu -m eval.eval"
        export CMD="--model hf --tasks \"$TASK\" --model_args \"trust_remote_code=True,pretrained=$MODEL\" --batch_size {cached_batch_size} --output_path logs"
        srun --ntasks=$NUM_NODES --export=ALL --wait=60 --kill-on-bad-exit=1 bash -c "$LAUNCHER $CMD"
"""
    else:
        multi_node_logic = ""
        if cached_batch_size is None:
            cached_batch_size = "auto"

        run_eval_inner = f"""        srun --ntasks=1 --export=ALL --wait=60 --kill-on-bad-exit=1 accelerate launch --num-processes "$GPUS_PER_NODE" --num-machines 1 \\
            --multi-gpu -m eval.eval \\
            --model hf \\
        --tasks "$TASK" \\
        --model_args "trust_remote_code=True,pretrained=$MODEL" \\
        --batch_size {cached_batch_size} \\
        --output_path logs
"""

    # Create a safe name for the job
    model_short = model.replace("/", "_").replace(" ", "_")
    job_name = f"eval_{model_short}_{task}"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/mv_exp/{model_short}/{task}_%j.%x.%N.out
#SBATCH --error=slurm_logs/mv_exp/{model_short}/{task}_%j.%x.%N.err
#SBATCH --time=00-12:00:00
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes={NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gres=gpu:{GPUS_PER_NODE}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com

# Resource configuration (can be tweaked per-script if needed)
NUM_NODES={NUM_NODES}
GPUS_PER_NODE={GPUS_PER_NODE}

{multi_node_logic}

export WORK_DIR=/e/project1/reformo/ali
export EVALCHEMY_DIR=$WORK_DIR/evalchemy
export TMPDIR=$WORK_DIR/.tmp
export TMP=$TMPDIR

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
module load Stages/2025
module load CUDA/12
source .venv/bin/activate

export SRUN_CPUS_PER_TASK={18 * GPUS_PER_NODE}

# 4. Run evaluation
for MODEL in "${{MODEL_NAMES[@]}}"; do
    for TASK in "${{TASKS[@]}}"; do
        echo "==========================================================="
        echo "Starting evaluation"
        echo "Model: $MODEL"
        echo "Task: $TASK"
        echo "==========================================================="
{run_eval_inner}    done
done
"""
    return script


def get_safe_filename(model: str, task: str) -> str:
    """Generate a safe filename for the script."""
    # Replace any problematic characters
    model = model.replace("/", "_").replace(" ", "_")
    return f"eval_{model}_{task}.sh"


def remove_scripts(eval_scripts_dir: Path):
    """Remove all scripts in the evaluation scripts directory."""
    if eval_scripts_dir.exists():
        for file in eval_scripts_dir.glob("*.sh"):
            file.unlink()


def _unique_models_preserve_order(model_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for mid in model_ids:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def cache_models_with_transformers(model_ids: list[str]) -> None:
    """Download each model and tokenizer into the local HF/Transformers cache."""
    unique = _unique_models_preserve_order(model_ids)
    n = len(unique)
    num_digits = len(str(n)) if n else 1
    print(f"Caching {n} unique model(s) via Transformers (tokenizer + weights)...")
    print()
    for i, model_id in enumerate(unique, start=1):
        print(f"[{i:0{num_digits}d}/{n}] Caching: {model_id}")
        try:
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(
                f"ERROR: failed to cache {model_id}: {e}\n"
                "If this is a private repo, set HF_TOKEN or HUGGING_FACE_HUB_TOKEN.",
                file=sys.stderr,
            )
            sys.exit(1)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SLURM evaluation scripts for each model-task pair.",
        epilog=(
            "By default, each unique model in MODELS is downloaded into the Hugging Face / "
            "Transformers cache (needed when compute nodes use TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE). "
            "Use --no-download-models to only regenerate shell scripts. "
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for private repositories."
        ),
    )
    parser.add_argument(
        "--no-download-models",
        action="store_true",
        help="Skip tokenizer and AutoModelForCausalLM cache warming; only write eval_scripts.",
    )
    parser.add_argument(
        "--skip-uncached-eval-scripts",
        action="store_true",
        help="Do not generate evaluation scripts for model/task pairs missing a cached batch size.",
    )
    return parser.parse_args()


def remove_finished_scripts(eval_scripts_dir: Path, results_dir: Path):
    """Remove all finished scripts in the evaluation scripts directory."""
    all_scripts = [f.name for f in eval_scripts_dir.glob("*.sh")]
    print(f"All scripts: {len(all_scripts)}")
    finished_scripts = get_finished_scripts(results_dir)
    print(f"Finished scripts: {len([f for f in all_scripts if f in finished_scripts])}")
    running_scripts = subprocess.check_output(
        ["squeue", "--me", "-h", "-o", "%j"], text=True
    )
    running_scripts = [
        s.strip() + ".sh" for s in running_scripts.split("\n") if s.strip()
    ]
    print(f"Running scripts: {len(running_scripts)}")
    remaining_scripts = [
        s for s in all_scripts if s not in finished_scripts and s not in running_scripts
    ]
    print(f"Remaining scripts: {len(remaining_scripts)}")

    for script in eval_scripts_dir.glob("*.sh"):
        if script.name not in remaining_scripts:
            script.unlink()


def get_finished_scripts(results_dir: Path) -> set[str]:
    finished_scripts: set[str] = set()
    for d in os.scandir(results_dir):
        tasks: list[str] = []
        for f in os.scandir(d):
            if not f.is_file():
                continue
            if open(f.path).read().strip() == "":
                continue
            result = json.load(open(f.path))
            for task, metrics in result["results"].items():
                if metrics == {}:
                    continue
                tasks.append(task)

        model = d.name.replace("__", "/")
        for task in tasks:
            finished_scripts.add(get_safe_filename(model, task))

    return finished_scripts


def main():
    args = parse_args()

    if not args.no_download_models:
        cache_models_with_transformers(MODELS)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    batch_size_cache_path = get_batch_size_cache_path(script_dir)
    batch_size_cache = load_batch_size_cache(batch_size_cache_path)
    batch_size_cache_path.touch(exist_ok=True)
    if batch_size_cache_path.stat().st_size == 0:
        batch_size_cache_path.write_text("{}\n")

    # Create the eval_scripts folder
    eval_scripts_dir = script_dir / "eval_scripts"
    eval_scripts_dir.mkdir(exist_ok=True)
    discovery_scripts_dir = script_dir / DISCOVERY_SCRIPTS_DIRNAME
    discovery_scripts_dir.mkdir(exist_ok=True)

    print(f"Creating evaluation scripts in: {eval_scripts_dir}")
    print(f"Creating discovery scripts in: {discovery_scripts_dir}")
    print(f"Models: {len(MODELS)}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Total scripts to generate: {len(MODELS) * len(TASKS)}")
    print()

    remove_scripts(eval_scripts_dir)
    remove_scripts(discovery_scripts_dir)
    finished_scripts = get_finished_scripts(Path("logs"))

    script_count = 0
    discovery_script_count = 0
    skipped_eval_script_count = 0

    for model in MODELS:
        for task in TASKS:
            filename = get_safe_filename(model, task)
            filepath = eval_scripts_dir / filename
            discovery_filename = f"discover_{filename}"
            discovery_filepath = discovery_scripts_dir / discovery_filename

            cached_batch_size = resolve_cached_batch_size(batch_size_cache, model, task)
            should_generate_eval = not (
                args.skip_uncached_eval_scripts and cached_batch_size is None
            )

            if should_generate_eval:
                script_content = generate_script(model, task, cached_batch_size)
                with open(filepath, "w") as f:
                    f.write(script_content)
                os.chmod(filepath, 0o755)
                script_count += 1
            else:
                skipped_eval_script_count += 1

            should_generate_discovery = (
                cached_batch_size is None and filename not in finished_scripts
            )

            if should_generate_discovery:
                discovery_script_content = generate_discovery_script(model, task)
                with open(discovery_filepath, "w") as f:
                    f.write(discovery_script_content)
                os.chmod(discovery_filepath, 0o755)
                discovery_script_count += 1

            n_scripts = len(MODELS) * len(TASKS)
            num_digits = len(str(n_scripts))
            progress = script_count + skipped_eval_script_count
            if not should_generate_eval and should_generate_discovery:
                print(
                    f"[{progress:0{num_digits}d}/{n_scripts}] Skipped: {filename} (missing cached batch size); generated {discovery_filename}"
                )
            elif not should_generate_eval:
                print(
                    f"[{progress:0{num_digits}d}/{n_scripts}] Skipped: {filename} (missing cached batch size; no discovery script needed)"
                )
            elif should_generate_discovery:
                print(
                    f"[{progress:0{num_digits}d}/{n_scripts}] Generated: {filename} and {discovery_filename}"
                )
            elif cached_batch_size is None and filename in finished_scripts:
                print(
                    f"[{progress:0{num_digits}d}/{n_scripts}] Generated: {filename} (finished eval; skipped {discovery_filename})"
                )
            else:
                print(
                    f"[{progress:0{num_digits}d}/{n_scripts}] Generated: {filename} (cached batch size; skipped {discovery_filename})"
                )

    print()
    print(
        f"Successfully generated {script_count} evaluation scripts in {eval_scripts_dir}"
    )
    print(
        f"Successfully generated {discovery_script_count} discovery scripts in {discovery_scripts_dir}"
    )
    print(
        f"Skipped {skipped_eval_script_count} evaluation scripts without cached batch sizes"
    )

    remove_finished_scripts(eval_scripts_dir, Path("logs"))


if __name__ == "__main__":
    main()
