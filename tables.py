import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


def collect_scores() -> pd.DataFrame:
    json_files = [
        f
        for d in os.scandir("logs")
        for f in os.scandir(d)
        if f.is_file()
        if open(f.path).read().strip() != ""
        and list(json.load(open(f.path))["results"].values())[0] != {}
    ]

    # Metadata fields to exclude (not actual scores)
    METADATA_FIELDS = {
        "num_examples",
        "completion_rate",
        "examples",
        "num_total",
        "solved_avg",
        "run_stats",
        "accuracy_std_err",
        "num_repeat",
        "accuracy_easy_avg",
        "accuracy_easy_std_err",
        "accuracy_medium_avg",
        "accuracy_medium_std_err",
        "accuracy_hard_avg",
        "accuracy_hard_std_err",
        "raw_metrics",
        "num_solved",
    }

    rows = []
    for json_file in json_files:
        with open(json_file.path, "r") as f:
            data = json.load(f)

        model = data.get("model_name", "unknown")
        results = data.get("results", {})

        for task, metrics in results.items():
            for metric, score in metrics.items():
                if metric not in METADATA_FIELDS:
                    rows.append(
                        {
                            "huggingface_model_id": model,
                            "benchmark": task,
                            "metric": metric,
                            "score": score,
                        }
                    )

    df = pd.DataFrame(rows)
    df = df.groupby(["huggingface_model_id", "benchmark", "metric"])
    df = df.agg({"score": "max"}).reset_index()
    df = df.groupby(["huggingface_model_id", "benchmark"])
    df = df.agg({"score": "mean"}).reset_index()
    df["score"] = df["score"] * 100.0
    df["score"] = df["score"].round(2)

    # save to csv
    df.to_csv("scores.csv", index=False)
    return df


BENCHMARKS_MAPPING = {
    "A24": "AIME24",
    "A25": "AIME25",
    "AMC": "AMC23",
    "M500": "MATH500",
    "HE": "HumanEval",
    "LCB": "LiveCodeBench",
    "MBPP": "MBPP",
    "GPQA": "GPQADiamond",
    "JEE": "JEEBench",
    "IFE": "IFEval",
}
BENCHMARKS_FULL_NAMES = list(BENCHMARKS_MAPPING.values())
DEFAULT_CUSTOM_CAPTION = "Selected models on reasoning benchmarks."
DEFAULT_CUSTOM_LABEL = "tab:evalchemy_custom"
DEFAULT_CUSTOM_GROUP_NAME = "Selected Models"
DIVIDER = "DIVIDER"


@dataclass(frozen=True)
class ModelRow:
    display_name: str
    hf_id: str
    section: str = "Main"

    @property
    def registry_name(self) -> str:
        return f"{self.display_name} [{self.section}]"


def model(section: str, display_name: str, hf_id: str) -> ModelRow:
    return ModelRow(display_name=display_name, hf_id=hf_id, section=section)


MODEL_REGISTRY = [
    model("Main", "OpenSci-NM", "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096"),
    model(
        "Main",
        "OpenSci-NM + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "OpenSci-NM + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-DPO-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "OpenSci-NM-16k",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-long_sft_16k",
    ),
    model(
        "Main",
        "OpenSci-NM-16k + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "OpenSci-NM-16k + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-16k-DPO-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "OpenSci-NM-16k-SFT-16k",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16384-rope_theta-1M-long_sft_16k",
    ),
    model(
        "Main",
        "OpenSci-NM-16k-SFT-16k + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "OpenSci-NM-16k-SFT-16k + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "FineWeb-Edu",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096",
    ),
    model(
        "Main",
        "FineWeb-Edu + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "FineWeb-Edu + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-DPO-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "FineWeb-Edu-16k",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
    ),
    model(
        "Main",
        "FineWeb-Edu-16k + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "FineWeb-Edu-16k + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("Main", "DCLM", "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096"),
    model(
        "Main",
        "DCLM + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "DCLM + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-DPO-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "DCLM-16k",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k",
    ),
    model(
        "Main",
        "DCLM-16k + SFT",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "DCLM-16k + DPO",
        "ali-elganzory/open-sci-ref-v0.02-1.7b-dclm-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("Main", "Comma0.1", "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate"),
    model(
        "Main",
        "Comma0.1 + SFT",
        "ali-elganzory/1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "Comma0.1 + DPO",
        "ali-elganzory/1.7b-Comma0.1-300BT-DPO-Tulu3-decontaminated",
    ),
    model("Main", "Comma0.1-16k", "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k"),
    model(
        "Main",
        "Comma0.1-16k + SFT",
        "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "Comma0.1-16k + DPO",
        "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("Main", "SmolLM2", "ali-elganzory/SmolLM2-1.7B-WithChatTemplate"),
    model(
        "Main",
        "SmolLM2 + SFT",
        "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "SmolLM2 + DPO",
        "ali-elganzory/SmolLM2-1.7B-DPO-Tulu3-decontaminated",
    ),
    model("Main", "SmolLM2-16k", "ali-elganzory/SmolLM2-1.7B-16k"),
    model(
        "Main",
        "SmolLM2-16k + SFT",
        "ali-elganzory/SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "SmolLM2-16k + DPO",
        "ali-elganzory/SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated",
    ),
    model("Main", "Qwen2.5", "Qwen/Qwen2.5-1.5B"),
    model(
        "Main",
        "Qwen2.5 + SFT",
        "ali-elganzory/Qwen2.5-1.5B-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "Qwen2.5 + DPO",
        "ali-elganzory/Qwen2.5-1.5B-DPO-Tulu3-decontaminated",
    ),
    model("Main", "Qwen3", "Qwen/Qwen3-1.7B-Base"),
    model(
        "Main",
        "Qwen3 + SFT",
        "ali-elganzory/Qwen3-1.7B-Base-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "Qwen3 + DPO",
        "ali-elganzory/Qwen3-1.7B-Base-DPO-Tulu3-decontaminated",
    ),
    model("Main", "MV", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated"),
    model(
        "Main",
        "MV + SFT",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "MV + DPO",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated",
    ),
    model("Main", "MV-16k", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k"),
    model(
        "Main",
        "MV-16k + SFT",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Main",
        "MV-16k + DPO",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated",
    ),
    model("Ablation", "MV-woinstruct", "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT"),
    model(
        "Ablation",
        "MV-woinstruct + SFT",
        "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-woinstruct + DPO",
        "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-DPO-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-woinstruct-16k",
        "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k",
    ),
    model(
        "Ablation",
        "MV-woinstruct-16k + SFT",
        "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-woinstruct-16k + DPO",
        "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("Ablation", "MV-woweb", "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT"),
    model(
        "Ablation",
        "MV-woweb + SFT",
        "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-woweb + DPO",
        "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-DPO-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-woweb-16k",
        "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k",
    ),
    model(
        "Ablation",
        "MV-woweb-16k + SFT",
        "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-woweb-16k + DPO",
        "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("Ablation", "MV", "ali-elganzory/1.7b-MixtureVitae-100BT"),
    model(
        "Ablation",
        "MV + SFT",
        "ali-elganzory/1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV + DPO",
        "ali-elganzory/1.7b-MixtureVitae-100BT-DPO-Tulu3-decontaminated",
    ),
    model("Ablation", "MV-16k", "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k"),
    model(
        "Ablation",
        "MV-16k + SFT",
        "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "Ablation",
        "MV-16k + DPO",
        "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model(
        "Not Decontaminated",
        "MV",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-WithChatTemplate",
    ),
    model(
        "Not Decontaminated",
        "MV + SFT",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-SFT-Tulu3",
    ),
    model(
        "Not Decontaminated",
        "MV + DPO",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-DPO-Tulu3",
    ),
    model(
        "Not Decontaminated",
        "MV-16k",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate",
    ),
    model(
        "Not Decontaminated",
        "MV-16k + SFT",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3",
    ),
    model(
        "Not Decontaminated",
        "MV-16k + DPO",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-DPO-Tulu3",
    ),
    model("0.4B", "Baguettotron", "ali-elganzory/Baguettotron"),
    model(
        "0.4B",
        "Baguettotron + SFT",
        "ali-elganzory/Baguettotron-SFT-Tulu3-decontaminated",
    ),
    model(
        "0.4B",
        "Baguettotron + DPO",
        "ali-elganzory/Baguettotron-DPO-Tulu3-decontaminated",
    ),
    model("0.4B", "Baguettotron 16k", "ali-elganzory/Baguettotron-longsft_16k"),
    model(
        "0.4B",
        "Baguettotron 16k + SFT",
        "ali-elganzory/Baguettotron-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "0.4B",
        "Baguettotron 16k + DPO",
        "ali-elganzory/Baguettotron-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("0.4B", "MV", "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096"),
    model(
        "0.4B",
        "MV + SFT",
        "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-SFT-Tulu3-decontaminated",
    ),
    model(
        "0.4B",
        "MV + DPO",
        "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-DPO-Tulu3-decontaminated",
    ),
    model(
        "0.4B",
        "MV-16k",
        "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k",
    ),
    model(
        "0.4B",
        "MV-16k + SFT",
        "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
    ),
    model(
        "0.4B",
        "MV-16k + DPO",
        "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
    ),
    model("Merged", "4+16k", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged"),
    model(
        "Merged",
        "4+16k + SFT",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-SFT-Tulu3-decontaminated",
    ),
    model(
        "Merged",
        "4+16k + DPO",
        "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-merged-DPO-Tulu3-decontaminated",
    ),
]

MODEL_LOOKUP = {
    (row.section, row.display_name): index for index, row in enumerate(MODEL_REGISTRY)
}


def registry_index(section: str, display_name: str) -> int:
    return MODEL_LOOKUP[(section, display_name)]


def model_ref(section: str, display_name: str, rendered_name: str | None = None):
    index = registry_index(section, display_name)
    if rendered_name is None:
        return index
    return (index, rendered_name)


TABLE1_GROUP_SPECS = [
    (
        "MV-woinstruct",
        [
            model_ref("Ablation", "MV-woinstruct"),
            model_ref("Ablation", "MV-woinstruct + SFT"),
            model_ref("Ablation", "MV-woinstruct + DPO"),
        ],
    ),
    (
        "MV-woinstruct-16k",
        [
            model_ref("Ablation", "MV-woinstruct-16k"),
            model_ref("Ablation", "MV-woinstruct-16k + SFT"),
            model_ref("Ablation", "MV-woinstruct-16k + DPO"),
        ],
    ),
    (
        "MV-woweb",
        [
            model_ref("Ablation", "MV-woweb"),
            model_ref("Ablation", "MV-woweb + SFT"),
            model_ref("Ablation", "MV-woweb + DPO"),
        ],
    ),
    (
        "MV-woweb-16k",
        [
            model_ref("Ablation", "MV-woweb-16k"),
            model_ref("Ablation", "MV-woweb-16k + SFT"),
            model_ref("Ablation", "MV-woweb-16k + DPO"),
        ],
    ),
    (
        "MV",
        [
            model_ref("Ablation", "MV"),
            model_ref("Ablation", "MV + SFT"),
            model_ref("Ablation", "MV + DPO"),
        ],
    ),
    (
        "MV-16k",
        [
            model_ref("Ablation", "MV-16k"),
            model_ref("Ablation", "MV-16k + SFT"),
            model_ref("Ablation", "MV-16k + DPO"),
        ],
    ),
]

TABLE2_GROUP_SPECS = [
    (
        "MV",
        [
            model_ref("Main", "MV"),
            model_ref("Main", "MV + SFT"),
            model_ref("Main", "MV + DPO"),
        ],
    ),
    (
        "MV-16k",
        [
            model_ref("Main", "MV-16k"),
            model_ref("Main", "MV-16k + SFT"),
            model_ref("Main", "MV-16k + DPO"),
        ],
    ),
    (
        "OpenSci-NM",
        [
            model_ref("Main", "OpenSci-NM"),
            model_ref("Main", "OpenSci-NM + SFT"),
            model_ref("Main", "OpenSci-NM + DPO"),
        ],
    ),
    (
        "OpenSci-NM-16k",
        [
            model_ref("Main", "OpenSci-NM-16k-SFT-16k", "OpenSci-NM-16k"),
            model_ref("Main", "OpenSci-NM-16k-SFT-16k + SFT", "OpenSci-NM-16k + SFT"),
            model_ref("Main", "OpenSci-NM-16k-SFT-16k + DPO", "OpenSci-NM-16k + DPO"),
        ],
    ),
    (
        "FineWeb-Edu",
        [
            model_ref("Main", "FineWeb-Edu"),
            model_ref("Main", "FineWeb-Edu + SFT"),
            model_ref("Main", "FineWeb-Edu + DPO"),
        ],
    ),
    (
        "FineWeb-Edu-16k",
        [
            model_ref("Main", "FineWeb-Edu-16k"),
            model_ref("Main", "FineWeb-Edu-16k + SFT"),
            model_ref("Main", "FineWeb-Edu-16k + DPO"),
        ],
    ),
    (
        "DCLM",
        [
            model_ref("Main", "DCLM"),
            model_ref("Main", "DCLM + SFT"),
            model_ref("Main", "DCLM + DPO"),
        ],
    ),
    (
        "Comma0.1",
        [
            model_ref("Main", "Comma0.1"),
            model_ref("Main", "Comma0.1 + SFT"),
            model_ref("Main", "Comma0.1 + DPO"),
        ],
    ),
    (
        "Comma0.1-16k",
        [
            model_ref("Main", "Comma0.1-16k"),
            model_ref("Main", "Comma0.1-16k + SFT"),
            model_ref("Main", "Comma0.1-16k + DPO"),
        ],
    ),
    (
        "SmolLM2",
        [
            model_ref("Main", "SmolLM2"),
            model_ref("Main", "SmolLM2 + SFT"),
            model_ref("Main", "SmolLM2 + DPO"),
        ],
    ),
    (
        "SmolLM2-16k",
        [
            model_ref("Main", "SmolLM2-16k"),
            model_ref("Main", "SmolLM2-16k + SFT"),
            model_ref("Main", "SmolLM2-16k + DPO"),
        ],
    ),
    (
        "Qwen2.5",
        [
            model_ref("Main", "Qwen2.5"),
            model_ref("Main", "Qwen2.5 + SFT"),
            model_ref("Main", "Qwen2.5 + DPO"),
        ],
    ),
    (
        "Qwen3",
        [
            model_ref("Main", "Qwen3"),
            model_ref("Main", "Qwen3 + SFT"),
            model_ref("Main", "Qwen3 + DPO"),
        ],
    ),
]

TABLE3_GROUP_SPECS = [
    (
        "Not Decontaminated",
        [
            model_ref("Not Decontaminated", "MV"),
            model_ref("Not Decontaminated", "MV + SFT"),
            model_ref("Not Decontaminated", "MV + DPO"),
            DIVIDER,
            model_ref("Not Decontaminated", "MV-16k"),
            model_ref("Not Decontaminated", "MV-16k + SFT"),
            model_ref("Not Decontaminated", "MV-16k + DPO"),
        ],
    ),
    (
        "Decontaminated",
        [
            model_ref("Main", "MV"),
            model_ref("Main", "MV + SFT"),
            model_ref("Main", "MV + DPO"),
            DIVIDER,
            model_ref("Main", "MV-16k"),
            model_ref("Main", "MV-16k + SFT"),
            model_ref("Main", "MV-16k + DPO"),
        ],
    ),
]

TABLE4_GROUP_SPECS = [
    (
        "Baguettotron",
        [
            model_ref("0.4B", "Baguettotron"),
            model_ref("0.4B", "Baguettotron + SFT"),
            model_ref("0.4B", "Baguettotron + DPO"),
        ],
    ),
    (
        "Baguettotron 16k",
        [
            model_ref("0.4B", "Baguettotron 16k"),
            model_ref("0.4B", "Baguettotron 16k + SFT"),
            model_ref("0.4B", "Baguettotron 16k + DPO"),
        ],
    ),
    (
        "MV (0.4B)",
        [
            model_ref("0.4B", "MV", "MV (0.4B)"),
            model_ref("0.4B", "MV + SFT", "MV (0.4B) + SFT"),
            model_ref("0.4B", "MV + DPO", "MV (0.4B) + DPO"),
        ],
    ),
    (
        "MV-16k (0.4B)",
        [
            model_ref("0.4B", "MV-16k", "MV-16k (0.4B)"),
            model_ref("0.4B", "MV-16k + SFT", "MV-16k (0.4B) + SFT"),
            model_ref("0.4B", "MV-16k + DPO", "MV-16k (0.4B) + DPO"),
        ],
    ),
]

FIXED_TABLE_SPECS = [
    (
        "Ablation on MixtureVitae mixture composition on reasoning benchmarks.",
        "tab:evalchemy_table1",
        TABLE1_GROUP_SPECS,
        False,
    ),
    (
        "MixtureVitae against other baselines on reasoning benchmarks (1.5B-1.7B scale).",
        "tab:evalchemy_table2",
        TABLE2_GROUP_SPECS,
        False,
    ),
    (
        "The effect of decontamination on the model's performance on reasoning benchmarks.",
        "tab:evalchemy_table3",
        TABLE3_GROUP_SPECS,
        True,
    ),
    (
        "Performance of 0.4B scale models on reasoning benchmarks.",
        "tab:evalchemy_table4",
        TABLE4_GROUP_SPECS,
        False,
    ),
]


def build_score_accessors(df: pd.DataFrame):
    if not df.empty:
        df_pivot = df.pivot(
            index="huggingface_model_id", columns="benchmark", values="score"
        )
    else:
        df_pivot = pd.DataFrame()

    def get_score(hf_id: str | None, benchmark: str):
        if hf_id and hf_id in df_pivot.index and benchmark in df_pivot.columns:
            value = df_pivot.loc[hf_id, benchmark]
            if pd.notna(value) and value != "-":
                return float(value)
        return "-"

    def get_scores_dict(hf_id: str):
        row_scores = {
            benchmark: get_score(hf_id, benchmark)
            for benchmark in BENCHMARKS_FULL_NAMES
        }
        avg_score = get_score(hf_id, "Avg")
        if avg_score == "-":
            valid_scores = [
                row_scores[benchmark]
                for benchmark in BENCHMARKS_FULL_NAMES
                if row_scores[benchmark] != "-"
            ]
            row_scores["Avg"] = (
                sum(valid_scores) / len(valid_scores) if valid_scores else "-"
            )
        else:
            row_scores["Avg"] = avg_score
        return row_scores

    return get_score, get_scores_dict


def resolve_group_models(group_specs):
    resolved_groups = []
    for group_name, entries in group_specs:
        resolved_models = []
        for entry in entries:
            if entry == DIVIDER:
                resolved_models.append((DIVIDER, None))
                continue
            if isinstance(entry, tuple):
                registry_idx, rendered_name = entry
            else:
                registry_idx = entry
                rendered_name = MODEL_REGISTRY[registry_idx].display_name
            row = MODEL_REGISTRY[registry_idx]
            resolved_models.append((rendered_name, row.hf_id))
        resolved_groups.append((group_name, resolved_models))
    return resolved_groups


def build_table_latex(caption, label, groups, get_scores_dict, is_multirow=False):
    max_scores = {benchmark: -float("inf") for benchmark in BENCHMARKS_FULL_NAMES + ["Avg"]}
    table_data = []

    for group_name, models in groups:
        for model_name, hf_id in models:
            if model_name == DIVIDER:
                table_data.append((group_name, DIVIDER, None))
                continue
            row_scores = get_scores_dict(hf_id)
            for benchmark, value in row_scores.items():
                if value != "-":
                    max_scores[benchmark] = max(max_scores[benchmark], value)
            table_data.append((group_name, model_name, row_scores))

    latex = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\scriptsize",
    ]

    if is_multirow:
        latex.extend(
            [
                r"\begin{tabular}{llccccccccccc}",
                r"\toprule",
                r"\textbf{Base} & \textbf{Model} & \textbf{A24} & \textbf{A25} & \textbf{AMC} & \textbf{M500} & \textbf{HE} & \textbf{LCB} & \textbf{MBPP} & \textbf{GPQA} & \textbf{JEE} & \textbf{IFE} & \textbf{Avg} \\",
            ]
        )
    else:
        latex.extend(
            [
                r"\begin{tabular}{lccccccccccc}",
                r"\toprule",
                r"\textbf{Model} & \textbf{A24} & \textbf{A25} & \textbf{AMC} & \textbf{M500} & \textbf{HE} & \textbf{LCB} & \textbf{MBPP} & \textbf{GPQA} & \textbf{JEE} & \textbf{IFE} & \textbf{Avg} \\",
            ]
        )

    latex.append(r"\midrule")

    current_group = None
    for group_name, model_name, row_scores in table_data:
        if current_group != group_name:
            if current_group is not None:
                latex.append(r"\midrule")
                latex.append(r"\addlinespace[0.3em]")
            current_group = group_name
            group_idx = 0

        if model_name == DIVIDER:
            latex.append(r"\cmidrule{2-13}" if is_multirow else r"\midrule")
            continue

        formatted_scores = []
        for benchmark in BENCHMARKS_FULL_NAMES + ["Avg"]:
            value = row_scores[benchmark]
            if value == "-":
                formatted_scores.append("--")
                continue
            value_str = f"{value:,.1f}"
            if value == max_scores[benchmark]:
                formatted_scores.append(f"\\textbf{{{value_str}}}")
            else:
                formatted_scores.append(value_str)
        scores_str = " & ".join(formatted_scores)

        if is_multirow:
            num_models = sum(
                1 for g, model, _ in table_data if g == group_name and model != DIVIDER
            )
            if group_idx == 0:
                latex.append(
                    f"\\multirow{{{num_models}}}{{*}}{{{group_name}}} & {model_name} & {scores_str} \\\\"
                )
            else:
                latex.append(f" & {model_name} & {scores_str} \\\\")
        else:
            latex.append(f"{model_name} & {scores_str} \\\\")

        group_idx += 1

    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(latex)


def escape_hf_id(hf_id: str) -> str:
    return (
        hf_id.replace("_", r"\_")
        .replace("-", r"-\allowbreak{}")
        .replace("/", r"/\allowbreak{}")
    )


def write_appendix(appendix_file: str, resolved_groups):
    extracted_models = {}
    for group_name, models in resolved_groups:
        for model_name, hf_id in models:
            if model_name != DIVIDER and hf_id is not None:
                extracted_models[model_name] = escape_hf_id(hf_id)

    appendix_latex = [
        r"\section{Appendix}",
        r"\subsection{Benchmark Details}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Benchmark Abbreviations and Full Names}",
        r"\label{tab:benchmark_mapping}",
        r"\scriptsize",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Abbreviation} & \textbf{Full Name} \\",
        r"\midrule",
    ]
    for abbrev, full_name in BENCHMARKS_MAPPING.items():
        appendix_latex.append(f"{abbrev} & {full_name} \\\\")
    appendix_latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\subsection{Model Mappings}",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Model Names to Hugging Face ID Mappings}",
            r"\label{tab:model_mapping}",
            r"\scriptsize",
            r"\begin{tabular}{lp{10cm}}",
            r"\toprule",
            r"\textbf{Model Name} & \textbf{Hugging Face ID} \\",
            r"\midrule",
        ]
    )
    for model_name in sorted(extracted_models):
        appendix_latex.append(
            f"{model_name} & \\texttt{{{extracted_models[model_name]}}} \\\\"
        )
    appendix_latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    Path(appendix_file).write_text("\n".join(appendix_latex) + "\n", encoding="utf-8")
    print(f"Appendix successfully written to {appendix_file}")


def write_missing_scores_csv(get_score, resolved_groups):
    missing_data = []
    for group_name, models in resolved_groups:
        for model_name, hf_id in models:
            if model_name == DIVIDER:
                continue
            identifier = hf_id if hf_id else model_name
            for benchmark in BENCHMARKS_FULL_NAMES:
                if get_score(hf_id, benchmark) == "-":
                    missing_data.append(
                        {
                            "model_id_or_name": identifier,
                            "missing_benchmark": benchmark,
                        }
                    )

    pd.DataFrame(missing_data).drop_duplicates().to_csv("missing_scores.csv", index=False)
    print("Missing benchmarks successfully written to missing_scores.csv")


def generate_and_save_latex(
    df: pd.DataFrame, tables_file="tables.tex", appendix_file="appendix.tex"
):
    get_score, get_scores_dict = build_score_accessors(df)
    resolved_groups = [resolve_group_models(specs) for _, _, specs, _ in FIXED_TABLE_SPECS]
    table_blocks = []
    for (caption, label, _, is_multirow), groups in zip(FIXED_TABLE_SPECS, resolved_groups):
        table_blocks.append(
            build_table_latex(
                caption,
                label,
                groups,
                get_scores_dict=get_scores_dict,
                is_multirow=is_multirow,
            )
        )

    Path(tables_file).write_text("\n\n\n".join(table_blocks) + "\n", encoding="utf-8")
    print(f"Tables successfully written to {tables_file}")

    fixed_table_groups = [group for groups in resolved_groups for group in groups]
    write_appendix(appendix_file, fixed_table_groups)
    write_missing_scores_csv(get_score, fixed_table_groups)


def parse_model_indices(spec: str, total_models: int) -> list[int]:
    if not spec or not spec.strip():
        raise ValueError("Model selection cannot be empty.")

    indices = []
    seen = set()
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(f"Invalid range '{token}': start must be <= end.")
            values = range(start, end + 1)
        else:
            values = [int(token)]

        for value in values:
            if value < 1 or value > total_models:
                raise ValueError(
                    f"Model index {value} is out of bounds. Valid range is 1-{total_models}."
                )
            zero_based = value - 1
            if zero_based not in seen:
                seen.add(zero_based)
                indices.append(zero_based)

    if not indices:
        raise ValueError("Model selection cannot be empty.")
    return indices


def print_model_registry():
    for index, row in enumerate(MODEL_REGISTRY, start=1):
        print(f"{index:>3}. {row.registry_name}: {row.hf_id}")


def select_indices_interactively() -> list[int]:
    if not sys.stdin.isatty():
        raise ValueError(
            "Interactive model selection requires a TTY. Pass --indices or use --list-models first."
        )

    print_model_registry()
    while True:
        raw_value = input(
            "Enter model indices/ranges (for example: 1,3,5-8): "
        ).strip()
        try:
            return parse_model_indices(raw_value, len(MODEL_REGISTRY))
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


def build_custom_groups(indices: list[int]):
    return [
        (
            DEFAULT_CUSTOM_GROUP_NAME,
            [
                (MODEL_REGISTRY[index].registry_name, MODEL_REGISTRY[index].hf_id)
                for index in indices
            ],
        )
    ]


def build_output_paths(output: str | None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = Path(output)
        if output_path.suffix:
            tex_path = output_path.with_suffix(".tex")
        else:
            tex_path = Path(f"{output_path}.tex")
    else:
        tex_path = Path("tables") / f"table_{timestamp}.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    return tex_path, tex_path.with_suffix(".pdf")


def build_standalone_document(table_latex: str) -> str:
    fixed_table_latex = table_latex.replace(r"\begin{table}[ht]", r"\begin{table}[H]", 1)
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=0.5in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{multirow}",
            r"\usepackage{float}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            fixed_table_latex,
            r"\end{document}",
            "",
        ]
    )


def export_table_pdf(table_latex: str, pdf_path: Path):
    standalone_document = build_standalone_document(table_latex)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tex_path = temp_path / "table.tex"
        tex_path.write_text(standalone_document, encoding="utf-8")
        try:
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(temp_path),
                    str(tex_path.name),
                ],
                cwd=temp_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("Warning: pdflatex is not available on PATH, so no PDF was generated.")
            return
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            stdout = exc.stdout.strip()
            message = stderr or stdout or "Unknown pdflatex error."
            print(f"Warning: failed to build PDF with pdflatex: {message}")
            return

        built_pdf = temp_path / "table.pdf"
        if not built_pdf.exists():
            print("Warning: pdflatex finished without producing a PDF.")
            return
        shutil.copy(built_pdf, pdf_path)
        print(f"Custom table PDF successfully written to {pdf_path}")


def write_custom_table_outputs(
    df: pd.DataFrame,
    indices: list[int],
    output: str | None = None,
    caption: str = DEFAULT_CUSTOM_CAPTION,
    label: str = DEFAULT_CUSTOM_LABEL,
):
    get_score, get_scores_dict = build_score_accessors(df)
    del get_score
    tex_path, pdf_path = build_output_paths(output)
    custom_groups = build_custom_groups(indices)
    custom_table_latex = build_table_latex(
        caption,
        label,
        custom_groups,
        get_scores_dict=get_scores_dict,
    )
    tex_path.write_text(custom_table_latex + "\n", encoding="utf-8")
    print(f"Custom table successfully written to {tex_path}")
    export_table_pdf(custom_table_latex, pdf_path)


def load_scores_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Evalchemy table, appendix, and custom table generator"
    )
    parser.add_argument(
        "--collect", action="store_true", help="Collect scores and save to scores.csv"
    )
    parser.add_argument(
        "--latex", action="store_true", help="Generate tables.tex and appendix.tex"
    )
    parser.add_argument(
        "--custom-table",
        action="store_true",
        help="Generate a custom table fragment and matching PDF.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the numbered global model registry used by --custom-table.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="1-based model indices or ranges, such as 1,3,5-8.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=DEFAULT_CUSTOM_CAPTION,
        help="Caption to use for --custom-table.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=DEFAULT_CUSTOM_LABEL,
        help="LaTeX label to use for --custom-table.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for --custom-table .tex file. Defaults to tables/table_<timestamp>.tex.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="scores.csv",
        help="Path to read scores CSV for --latex and --custom-table.",
    )

    args = parser.parse_args()

    did_work = False
    df = None

    if args.list_models:
        print_model_registry()
        did_work = True

    if args.collect:
        df = collect_scores()
        did_work = True

    if args.latex:
        if df is None:
            df = load_scores_dataframe(args.csv)
        generate_and_save_latex(df)
        did_work = True

    if args.custom_table:
        if df is None:
            df = load_scores_dataframe(args.csv)
        try:
            indices = (
                parse_model_indices(args.indices, len(MODEL_REGISTRY))
                if args.indices
                else select_indices_interactively()
            )
        except ValueError as exc:
            parser.error(str(exc))
        write_custom_table_outputs(
            df,
            indices,
            output=args.output,
            caption=args.caption,
            label=args.label,
        )
        did_work = True

    if not did_work:
        print(
            "Nothing to do: pass --collect, --latex, --custom-table, --list-models, or a combination."
        )


if __name__ == "__main__":
    main()
