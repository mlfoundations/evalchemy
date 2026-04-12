import pandas as pd
import os
import json


def collect_scores() -> pd.DataFrame:
    json_files = [
        f
        for d in os.scandir("logs")
        for f in os.scandir(d)
        if f.is_file()
        if list(json.load(open(f.path))["results"].values())[0] != {}
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
    df = df.groupby(["huggingface_model_id", "benchmark"])
    df = df.agg({"score": "mean"}).reset_index()
    df["score"] = df["score"] * 100.0
    df["score"] = df["score"].round(2)

    # save to csv
    df.to_csv("scores.csv", index=False)
    return df


def generate_and_save_latex(
    df: pd.DataFrame, tables_file="tables.tex", appendix_file="appendix.tex"
):
    # Pivot the dataframe for fast lookups: huggingface_model_id -> benchmark -> score
    if not df.empty:
        df_pivot = df.pivot(
            index="huggingface_model_id", columns="benchmark", values="score"
        )
    else:
        df_pivot = pd.DataFrame()

    benchmarks_mapping = {
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

    benchmarks_full_names = list(benchmarks_mapping.values())

    def get_score(hf_id, b):
        if hf_id and hf_id in df_pivot.index and b in df_pivot.columns:
            val = df_pivot.loc[hf_id, b]
            if pd.notna(val) and val != "-":
                return float(val)
        return "-"

    def get_scores_dict(hf_id):
        row = {}
        for b in benchmarks_full_names:
            row[b] = get_score(hf_id, b)

        # Extract or compute average if not explicitly provided
        avg_score = get_score(hf_id, "Avg")
        if avg_score == "-":
            valid_scores = [row[b] for b in benchmarks_full_names if row[b] != "-"]
            row["Avg"] = sum(valid_scores) / len(valid_scores) if valid_scores else "-"
        else:
            row["Avg"] = avg_score
        return row

    # Mapping based on the provided status list
    # Format: (Group Name, [(Model Name, HuggingFace ID), ...])
    # Note: Use ("DIVIDER", None) to inject an intra-group separator.

    table1_groups = [
        (
            "MV-noinstruct",
            [
                ("MV-noinstruct", "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT"),
                (
                    "MV-noinstruct + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV-noinstruct + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-web_curated-100BT-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "MV-noinstruct-16k",
            [
                (
                    "MV-noinstruct-16k",
                    "ontocord/1.7b-MixtureVitae-web_curated-100BT-longsft_16k",
                ),
                ("MV-noinstruct-16k + SFT", None),
                ("MV-noinstruct-16k + DPO", None),
            ],
        ),
        (
            "MV-noweb",
            [
                ("MV-noweb", "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT"),
                (
                    "MV-noweb + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV-noweb + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "MV-noweb-16k",
            [
                (
                    "MV-noweb-16k",
                    "ontocord/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k",
                ),
                (
                    "MV-noweb-16k + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV-noweb-16k + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-curated_instruct-100BT-longsft_16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "MV",
            [
                ("MV", "ali-elganzory/1.7b-MixtureVitae-100BT"),
                (
                    "MV + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-100BT-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-100BT-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "MV-16k",
            [
                ("MV-16k", "ontocord/1.7b-MixtureVitae-100BT-longsft_16k"),
                (
                    "MV-16k + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV-16k + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-100BT-longsft_16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
    ]

    table2_groups = [
        (
            "MV",
            [
                ("MV", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated"),
                (
                    "MV + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "MV-16k",
            [
                (
                    "MV-16k",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
                ),
                (
                    "MV-16k + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV-16k + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "OpenSci-NM",
            [
                (
                    "OpenSci-NM",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096",
                ),
                (
                    "OpenSci-NM + SFT",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-SFT-Tulu3-decontaminated",
                ),
                (
                    "OpenSci-NM + DPO",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-4096-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "OpenSci-NM-16k",
            [
                (
                    "OpenSci-NM-16k",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16384-rope_theta-1M-long_sft_16k",
                ),
                (
                    "OpenSci-NM-16k + SFT",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "OpenSci-NM-16k + DPO",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-nemotron-hq-300B-16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "FineWeb-Edu",
            [
                (
                    "FineWeb-Edu",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096",
                ),
                (
                    "FineWeb-Edu + SFT",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-SFT-Tulu3-decontaminated",
                ),
                (
                    "FineWeb-Edu + DPO",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "FineWeb-Edu-16k",
            [
                (
                    "FineWeb-Edu-16k",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-4096-longsft_16k",
                ),
                (
                    "FineWeb-Edu-16k + SFT",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "FineWeb-Edu-16k + DPO",
                    "ali-elganzory/open-sci-ref-v0.02-1.7b-fineweb-edu-1.4t-300B-4096-longsft_16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "Comma0.1",
            [
                ("Comma0.1", "ali-elganzory/1.7b-Comma0.1-300BT-WithChatTemplate"),
                (
                    "Comma0.1 + SFT",
                    "ali-elganzory/1.7b-Comma0.1-300BT-SFT-Tulu3-decontaminated",
                ),
                (
                    "Comma0.1 + DPO",
                    "ali-elganzory/1.7b-Comma0.1-300BT-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "Comma0.1-16k",
            [
                ("Comma0.1-16k", "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k"),
                (
                    "Comma0.1-16k + SFT",
                    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "Comma0.1-16k + DPO",
                    "ali-elganzory/1.7b-Comma0.1-300BT-longsft_16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "SmolLM2",
            [
                ("SmolLM2", "ali-elganzory/SmolLM2-1.7B-WithChatTemplate"),
                (
                    "SmolLM2 + SFT",
                    "ali-elganzory/SmolLM2-1.7B-SFT-Tulu3-decontaminated",
                ),
                (
                    "SmolLM2 + DPO",
                    "ali-elganzory/SmolLM2-1.7B-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "SmolLM2-16k",
            [
                ("SmolLM2-16k", "ontocord/SmolLM2-1.7B-16k"),
                (
                    "SmolLM2-16k + SFT",
                    "ali-elganzory/SmolLM2-1.7B-16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "SmolLM2-16k + DPO",
                    "ali-elganzory/SmolLM2-1.7B-16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "Qwen2.5",
            [
                ("Qwen2.5", "Qwen/Qwen2.5-1.5B"),
                (
                    "Qwen2.5 + SFT",
                    "ali-elganzory/Qwen2.5-1.5B-SFT-Tulu3-decontaminated",
                ),
                (
                    "Qwen2.5 + DPO",
                    "ali-elganzory/Qwen2.5-1.5B-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "Qwen3",
            [
                ("Qwen3", "Qwen/Qwen3-1.7B-Base"),
                (
                    "Qwen3 + SFT",
                    "ali-elganzory/Qwen3-1.7B-Base-SFT-Tulu3-decontaminated",
                ),
                (
                    "Qwen3 + DPO",
                    "ali-elganzory/Qwen3-1.7B-Base-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
    ]

    table3_groups = [
        (
            "Not Decontaminated",
            [
                ("MV", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-WithChatTemplate"),
                ("MV + SFT", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-SFT-Tulu3"),
                ("MV + DPO", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-DPO-Tulu3"),
                ("DIVIDER", None),
                (
                    "MV-16k",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-WithChatTemplate",
                ),
                (
                    "MV-16k + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-SFT-Tulu3",
                ),
                (
                    "MV-16k + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-16k-DPO-Tulu3",
                ),
            ],
        ),
        (
            "Decontaminated",
            [
                ("MV", "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated"),
                (
                    "MV + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-DPO-Tulu3-decontaminated",
                ),
                ("DIVIDER", None),
                (
                    "MV-16k",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k",
                ),
                (
                    "MV-16k + SFT",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV-16k + DPO",
                    "ali-elganzory/1.7b-MixtureVitae-300BT-v1-decontaminated-16k-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
    ]

    table4_groups = [
        (
            "Baguettotron",
            [
                ("Baguettotron", "PleIAs/Baguettotron"),
                (
                    "Baguettotron + SFT",
                    "ali-elganzory/Baguettotron-SFT-Tulu3-decontaminated",
                ),
                (
                    "Baguettotron + DPO",
                    "ali-elganzory/Baguettotron-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "Baguettotron 16k",
            [
                ("Baguettotron 16k", "ontocord/Baguettotron-longsft_16k"),
                ("Baguettotron 16k + SFT", None),
                ("Baguettotron 16k + DPO", None),
            ],
        ),
        (
            "MV (0.4B)",
            [
                (
                    "MV (0.4B)",
                    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096",
                ),
                (
                    "MV (0.4B) + SFT",
                    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-SFT-Tulu3-decontaminated",
                ),
                (
                    "MV (0.4B) + DPO",
                    "ali-elganzory/0.4b-mixturevitae-v1-decontaminated-300B-4096-DPO-Tulu3-decontaminated",
                ),
            ],
        ),
        (
            "MV-16k (0.4B)",
            [
                (
                    "MV-16k (0.4B)",
                    "ontocord/0.4b-mixturevitae-v1-decontaminated-300B-4096-longsft_16k",
                ),
                ("MV-16k (0.4B) + SFT", None),
                ("MV-16k (0.4B) + DPO", None),
            ],
        ),
    ]

    # --- Core table builder ---
    def build_table_latex(caption, label, groups, is_multirow=False):
        max_scores = {b: -float("inf") for b in benchmarks_full_names + ["Avg"]}
        table_data = []

        for group_name, models in groups:
            for model_name, hf_id in models:
                if model_name == "DIVIDER":
                    table_data.append((group_name, "DIVIDER", None))
                    continue

                row_scores = get_scores_dict(hf_id)
                for b, v in row_scores.items():
                    if v != "-":
                        max_scores[b] = max(max_scores[b], v)
                table_data.append((group_name, model_name, row_scores))

        latex = []
        latex.append(r"\begin{table}[ht]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append(r"\scriptsize")

        if is_multirow:
            latex.append(r"\begin{tabular}{llccccccccccc}")
            latex.append(r"\toprule")
            latex.append(
                r"\textbf{Base} & \textbf{Model} & \textbf{A24} & \textbf{A25} & \textbf{AMC} & \textbf{M500} & \textbf{HE} & \textbf{LCB} & \textbf{MBPP} & \textbf{GPQA} & \textbf{JEE} & \textbf{IFE} & \textbf{Avg} \\"
            )
        else:
            latex.append(r"\begin{tabular}{lccccccccccc}")
            latex.append(r"\toprule")
            latex.append(
                r"\textbf{Model} & \textbf{A24} & \textbf{A25} & \textbf{AMC} & \textbf{M500} & \textbf{HE} & \textbf{LCB} & \textbf{MBPP} & \textbf{GPQA} & \textbf{JEE} & \textbf{IFE} & \textbf{Avg} \\"
            )

        latex.append(r"\midrule")

        current_group = None
        for i, (group_name, model_name, row_scores) in enumerate(table_data):
            if current_group != group_name:
                if current_group is not None:
                    latex.append(r"\midrule")
                    latex.append(r"\addlinespace[0.3em]")
                current_group = group_name
                group_idx = 0

            if model_name == "DIVIDER":
                if is_multirow:
                    latex.append(r"\cmidrule{2-13}")
                else:
                    latex.append(r"\midrule")
                continue

            formatted_scores = []
            for b in benchmarks_full_names + ["Avg"]:
                v = row_scores[b]
                if v == "-":
                    formatted_scores.append("--")
                else:
                    # Apply commas for thousands readability
                    v_str = f"{v:,.1f}"
                    if v == max_scores[b]:
                        formatted_scores.append(f"\\textbf{{{v_str}}}")
                    else:
                        formatted_scores.append(v_str)

            scores_str = " & ".join(formatted_scores)

            if is_multirow:
                num_models = sum(
                    1 for g, m, _ in table_data if g == group_name and m != "DIVIDER"
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

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    # --- Build Tables File ---
    table1 = build_table_latex(
        "Ablation on MixtureVitae mixture composition on reasoning benchmarks.",
        "tab:evalchemy_table1",
        table1_groups,
    )

    table2 = build_table_latex(
        "MixtureVitae against other baselines on reasoning benchmarks (1.5B-1.7B scale).",
        "tab:evalchemy_table2",
        table2_groups,
    )

    table3 = build_table_latex(
        "The effect of decontamination on the model's performance on reasoning benchmarks.",
        "tab:evalchemy_table3",
        table3_groups,
        is_multirow=True,
    )

    table4 = build_table_latex(
        "Performance of 0.4B scale models on reasoning benchmarks.",
        "tab:evalchemy_table4",
        table4_groups,
    )

    tables_content = f"{table1}\n\n\n{table2}\n\n\n{table3}\n\n\n{table4}\n"

    with open(tables_file, "w") as f:
        f.write(tables_content)
    print(f"Tables successfully written to {tables_file}")

    # --- Build Appendix File ---
    # Extract unique models
    extracted_models = {}
    all_groups_lists = [table1_groups, table2_groups, table3_groups, table4_groups]

    for group_list in all_groups_lists:
        for group_name, models in group_list:
            for model_name, hf_id in models:
                if hf_id is not None and model_name != "DIVIDER":
                    # Escape underscores and add \allowbreak{} after hyphens and slashes
                    escaped_hf_id = hf_id.replace("_", r"\_").replace("-", r"-\allowbreak{}").replace("/", r"/\allowbreak{}")
                    extracted_models[model_name] = escaped_hf_id

    appendix_latex = []
    appendix_latex.append(r"\section{Appendix}")

    # 1. Benchmark Mapping Table
    appendix_latex.append(r"\subsection{Benchmark Details}")
    appendix_latex.append(r"\begin{table}[ht]")
    appendix_latex.append(r"\centering")
    appendix_latex.append(r"\caption{Benchmark Abbreviations and Full Names}")
    appendix_latex.append(r"\label{tab:benchmark_mapping}")
    appendix_latex.append(r"\scriptsize")
    appendix_latex.append(r"\begin{tabular}{ll}")
    appendix_latex.append(r"\toprule")
    appendix_latex.append(r"\textbf{Abbreviation} & \textbf{Full Name} \\")
    appendix_latex.append(r"\midrule")
    for abbrev, full_name in benchmarks_mapping.items():
        appendix_latex.append(f"{abbrev} & {full_name} \\\\")
    appendix_latex.append(r"\bottomrule")
    appendix_latex.append(r"\end{tabular}")
    appendix_latex.append(r"\end{table}")
    appendix_latex.append("")

    # 2. Model Mapping Table
    appendix_latex.append(r"\subsection{Model Mappings}")
    appendix_latex.append(r"\begin{table}[ht]")
    appendix_latex.append(r"\centering")
    appendix_latex.append(r"\caption{Model Names to Hugging Face ID Mappings}")
    appendix_latex.append(r"\label{tab:model_mapping}")
    appendix_latex.append(r"\scriptsize")
    # Using p column for Hugging Face IDs to prevent bleeding off the page
    appendix_latex.append(r"\begin{tabular}{lp{10cm}}")
    appendix_latex.append(r"\toprule")
    appendix_latex.append(r"\textbf{Model Name} & \textbf{Hugging Face ID} \\")
    appendix_latex.append(r"\midrule")

    # Sort alphabetically by model name for easier reading in the appendix
    for model_name in sorted(extracted_models.keys()):
        appendix_latex.append(
            f"{model_name} & \\texttt{{{extracted_models[model_name]}}} \\\\"
        )

    appendix_latex.append(r"\bottomrule")
    appendix_latex.append(r"\end{tabular}")
    appendix_latex.append(r"\end{table}")

    appendix_content = "\n".join(appendix_latex) + "\n"

    with open(appendix_file, "w") as f:
        f.write(appendix_content)
    print(f"Appendix successfully written to {appendix_file}")

    # --- 3. Build Missing Scores CSV ---
    missing_data = []
    for group_list in all_groups_lists:
        for group_name, models in group_list:
            for model_name, hf_id in models:
                if model_name != "DIVIDER":
                    identifier = hf_id if hf_id else model_name
                    for b in benchmarks_full_names:
                        if get_score(hf_id, b) == "-":
                            missing_data.append({
                                "model_id_or_name": identifier,
                                "missing_benchmark": b
                            })
                            
    missing_df = pd.DataFrame(missing_data).drop_duplicates()
    missing_df.to_csv("missing_scores.csv", index=False)
    print("Missing benchmarks successfully written to missing_scores.csv")


# Sample: regenerate tables and/or collect scores from logs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evalchemy table and appendix generator"
    )
    parser.add_argument(
        "--collect", action="store_true", help="Collect scores and save to scores.csv"
    )
    parser.add_argument(
        "--latex", action="store_true", help="Generate tables.tex and appendix.tex"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="scores.csv",
        help="Path to read scores CSV for --latex",
    )

    args = parser.parse_args()

    df = None
    if args.collect:
        df = collect_scores()
    if args.latex:
        if df is None:
            # Read scores from csv if not already collected in this run
            df = pd.read_csv(args.csv) if os.path.exists(args.csv) else pd.DataFrame()
        generate_and_save_latex(df)
    if not args.collect and not args.latex:
        print("Nothing to do: pass --collect, --latex, or both.")