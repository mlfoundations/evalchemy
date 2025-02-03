import pandas as pd
import numpy as np
import datetime
import argparse
import os
from glob import glob
from tqdm import tqdm

from eval.chat_benchmarks.arena_hard_auto.utils import load_model_answers
from eval.chat_benchmarks.arena_hard_auto.utils_math import (
    compute_mle_elo,
    get_bootstrap_result,
    get_win_rate_column,
    fit_bt,
    construct_style_matrices,
    get_bootstrap_result_style_control,
    STYLE_CONTROL_ELEMENTS,
    LENGTH_CONTROL_ELEMENTS,
    MARKDOWN_CONTROL_ELEMENTS,
)


def get_battles_from_row(row, first_game_only, multiplier, baseline_model, metadata=None):
    results = []
    output = {"question_id": row["question_id"], "model_a": baseline_model, "model_b": row["model"]}

    game = row["games"][0]
    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_a"
    elif game["score"] == "A>>B":
        output["winner"] = "model_a"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_b"
    elif game["score"] == "B>>A":
        output["winner"] = "model_b"
        weight = multiplier
    else:
        weight = 0

    # add conv_metadata for style control
    if metadata:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata[baseline_model][row["question_id"]]["conv_metadata"]["token_len"],
            "sum_assistant_b_tokens": metadata[row["model"]][row["question_id"]]["conv_metadata"]["token_len"],
            "header_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["header_count"],
            "header_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["header_count"],
            "list_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["list_count"],
            "list_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["list_count"],
            "bold_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["bold_count"],
            "bold_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["bold_count"],
        }

    if weight:
        results += [output] * weight

    if first_game_only:
        return results

    # game 2
    output = {"question_id": row["question_id"], "model_a": baseline_model, "model_b": row["model"]}

    game = row["games"][1]

    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_b"
    elif game["score"] == "A>>B":
        output["winner"] = "model_b"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_a"
    elif game["score"] == "B>>A":
        output["winner"] = "model_a"
        weight = multiplier
    else:
        weight = 0

    if metadata:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata[baseline_model][row["question_id"]]["conv_metadata"]["token_len"],
            "sum_assistant_b_tokens": metadata[row["model"]][row["question_id"]]["conv_metadata"]["token_len"],
            "header_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["header_count"],
            "header_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["header_count"],
            "list_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["list_count"],
            "list_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["list_count"],
            "bold_count_a": metadata[baseline_model][row["question_id"]]["conv_metadata"]["bold_count"],
            "bold_count_b": metadata[row["model"]][row["question_id"]]["conv_metadata"]["bold_count"],
        }

    if weight:
        results += [output] * weight

    return results


def get_battles_from_judgment(
    bench_name, judge_name, first_game_only=False, multiplier=3, baseline_model="gpt-4-0314", style_control=False
):
    print("Turning judgment results into battles...")

    judge_dir = f"eval/chat_benchmarks/arena_hard_auto/data/{bench_name}/model_judgment/{judge_name}"
    assert os.path.exists(judge_dir)

    judgments = pd.concat([pd.read_json(file, lines=True) for file in tqdm(glob(f"{judge_dir}/*jsonl"))])

    metadata = None
    if style_control:
        ans_dir = f"eval/chat_benchmarks/arena_hard_auto/data/{bench_name}/model_answer"
        assert os.path.exists(ans_dir)

        metadata = {}
        for file in tqdm(glob(f"{ans_dir}/*.jsonl")):
            df = pd.read_json(file, lines=True)
            assert (
                "conv_metadata" in df.columns
            ), "You must have conv_metadata attributes in your model answer to apply style contro. Please pull newest data if needed."
            metadata[df.model_id[0]] = df[["question_id", "conv_metadata"]].set_index("question_id").to_dict("index")

    battles = judgments.apply(
        lambda row: get_battles_from_row(row, first_game_only, multiplier, baseline_model, metadata), axis=1
    )
    battles = pd.DataFrame(battles[battles.map(len) > 0].explode().tolist())
    battles.to_json("eval/chat_benchmarks/arena_hard_auto/data/arena_hard_battles.jsonl", orient="records", lines=True)
    return battles


def generate_arena_hard_leaderboard(
    bench_name="arena-hard-v0.1",
    judge_name="gpt-4-1106-preview",
    baseline_model="gpt-4-0314",
    first_game_only=False,
    multiplier=3,
    num_rounds=100,
    show_elo=False,
    style_control=False,
    length_control_only=False,
    markdown_control_only=False,
):
    """
    Generate an Arena Hard leaderboard with various configuration options.

    Args:
        bench_name (str): Name of the benchmark dataset
        judge_name (str): Name of the judge model
        baseline_model (str): Baseline model for comparisons
        first_game_only (bool): Whether to use only the first game
        multiplier (int): Weight multiplier for decisive wins
        num_rounds (int): Number of bootstrap rounds
        show_elo (bool): Whether to show raw Elo scores
        style_control (bool): Apply style control in analysis
        length_control_only (bool): Control for response length only
        markdown_control_only (bool): Control for markdown formatting only

    Returns:
        pd.DataFrame: Leaderboard statistics
    """
    # Validate input
    assert (
        sum([style_control, length_control_only, markdown_control_only]) < 2
    ), "You can only control one of the three: length, markdown, or both style."

    # Load model answers
    answer_dir = os.path.join("data", bench_name, "model_answer")
    model_answers = load_model_answers(answer_dir)

    # Get battles from judgments
    battles = get_battles_from_judgment(
        bench_name,
        judge_name,
        first_game_only,
        multiplier,
        baseline_model,
        style_control or length_control_only or markdown_control_only,
    )

    # Perform analysis based on control type
    if style_control:
        X, Y, models = construct_style_matrices(battles)
        bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=baseline_model)
        bootstrap_model_coef, _ = get_bootstrap_result_style_control(
            X, Y, battles, models, fit_bt, num_round=num_rounds, baseline_model=baseline_model
        )
        display_coefs = {
            STYLE_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(STYLE_CONTROL_ELEMENTS) // 2)
        }
        print(f"Style Coefficients: {display_coefs}")
    elif length_control_only:
        X, Y, models = construct_style_matrices(battles, apply_ratio=[1], style_elements=LENGTH_CONTROL_ELEMENTS)
        bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=baseline_model)
        bootstrap_model_coef, _ = get_bootstrap_result_style_control(
            X, Y, battles, models, fit_bt, num_round=num_rounds, baseline_model=baseline_model
        )
        display_coefs = {
            LENGTH_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(LENGTH_CONTROL_ELEMENTS) // 2)
        }
        print(f"Style Coefficients: {display_coefs}")
    elif markdown_control_only:
        X, Y, models = construct_style_matrices(
            battles, apply_ratio=[1, 1, 1], style_elements=MARKDOWN_CONTROL_ELEMENTS
        )
        bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=baseline_model)
        bootstrap_model_coef, _ = get_bootstrap_result_style_control(
            X, Y, battles, models, fit_bt, num_round=num_rounds, baseline_model=baseline_model
        )
        display_coefs = {
            MARKDOWN_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(MARKDOWN_CONTROL_ELEMENTS) // 2)
        }
        print(f"Style Coefficients: {display_coefs}")
    else:
        bt_model_coef = compute_mle_elo(battles, baseline_model=baseline_model)
        bootstrap_model_coef = get_bootstrap_result(battles, compute_mle_elo, num_rounds, baseline_model)

    # Prepare stats DataFrame
    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bt_model_coef.index):
        assert model in bootstrap_model_coef.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bt_model_coef[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_model_coef[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_model_coef[model], 97.5)

        # Calculate average tokens
        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                if "token_len" in turn:
                    length += turn["token_len"]
                else:
                    length += row["conv_metadata"]["token_len"]
            length /= len(model_answers[model])

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_model_coef[model].tolist()

    # Convert to win rate if not showing raw Elo
    if not show_elo:
        stats.sort_values(by="model", inplace=True)
        stats["score"] = get_win_rate_column(stats, "score", baseline_model).tolist()
        stats["lower"] = get_win_rate_column(stats, "lower", baseline_model).tolist()
        stats["upper"] = get_win_rate_column(stats, "upper", baseline_model).tolist()
        decimal = 1
    else:
        decimal = 0
        stats = stats.astype({"score": int, "lower": int, "upper": int})

    # Sort and prepare for output
    stats.sort_values(by="score", ascending=False, inplace=True)

    # Prepare final DataFrame for CSV export
    cur_date = datetime.datetime.now()
    date_str = cur_date.strftime("%Y%m%d")
    stats_export = stats.drop(columns=["results"])

    # Compute confidence intervals
    CI = []
    for i in range(len(stats_export)):
        score = stats_export.iloc[i]["score"]
        upper = stats_export.iloc[i]["upper"]
        lower = stats_export.iloc[i]["lower"]
        CI.append(f"(-{(score-lower):.2f}, +{(upper-score):.2f})")

    stats_export["CI"] = CI
    stats_export.rename(columns={"upper": "rating_q975"}, inplace=True)
    stats_export.rename(columns={"lower": "rating_q025"}, inplace=True)

    # Reorder columns
    col_list = list(stats_export)
    col_list[-2], col_list[-1] = col_list[-1], col_list[-2]
    stats_export = stats_export.loc[:, col_list]

    # Add date
    stats_export["date"] = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]

    return stats_export
