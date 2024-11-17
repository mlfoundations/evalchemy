import os
import json
from tabulate import tabulate
from datetime import datetime
import csv
import io
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy import stats
from prettytable import PrettyTable
import argparse


def read_score_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_model_name(filename):
    name_dict = {"gpt-3": "gpt-3.5", "Meta-Llama-3": "llama-3.1-405b-instruct"}
    name_out = filename.split("_", 1)[1].split(".")[0]
    return name_dict.get(name_out, name_out)


def generate_csv(headers, table_data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(table_data)
    return output.getvalue()


def plot_overall_results(all_scores, source_dir):
    # Extract unique tested models and judge models
    tested_models = sorted(set(score["tested_model"] for score in all_scores))
    judge_models = sorted(set(score["judge_model"] for score in all_scores))

    if len(judge_models) < 2:
        print("Error: At least two different judge models are required for comparison.")
        return

    # Prepare data for plotting
    data = {tested: {} for tested in tested_models}
    for score in all_scores:
        tested = score["tested_model"]
        judge = score["judge_model"]
        overall_score = score.get("overall")
        data[tested][judge] = overall_score

    # Generate all possible pairs of judge models
    judge_pairs = list(combinations(judge_models, 2))

    # Set up the plot grid
    n_pairs = len(judge_pairs)
    n_cols = 2 if n_pairs > 2 else 1
    n_rows = (n_pairs + 1) // 2  # Round up division
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows))
    fig.suptitle("Comparison of Overall Scores by Different Judge Models", fontsize=16)

    # Flatten axs if it's a 2D array
    if n_pairs == 1:
        axs = [[axs]]

    # Colors for tested models
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tested_models)))

    for i, (judge1, judge2) in enumerate(judge_pairs):
        ai, aj = divmod(i, n_cols)
        ax = axs[ai][aj]  # Use indexing instead of conditional assignment

        # Collect valid data points for correlation calculation
        x_values = []
        y_values = []

        # Plot scatter points
        for tested, color in zip(tested_models, colors):
            if judge1 in data[tested] and judge2 in data[tested]:
                x = data[tested][judge1]
                y = data[tested][judge2]
                ax.scatter(x, y, c=[color], label=tested, s=100)
                ax.annotate(tested, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
                x_values.append(x)
                y_values.append(y)

        # Calculate correlation coefficient
        if len(x_values) > 1:  # Need at least two points for correlation
            corr, _ = stats.pearsonr(x_values, y_values)
            ax.text(
                0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes, verticalalignment="top", fontsize=10
            )

        # Customize the plot
        ax.set_xlabel(f"Score by {judge1}")
        ax.set_ylabel(f"Score by {judge2}")
        ax.set_title(f"{judge1} vs {judge2}")

        # Add a diagonal line
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="gray")

        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Add a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Tested Models", bbox_to_anchor=(1.05, 0.5), loc="center left")

    # Remove any unused subplots
    if n_pairs < len(axs):
        for j in range(n_pairs, len(axs)):
            fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    plt.savefig(f"{source_dir}/overall_scores_comparison.png", bbox_inches="tight")
    print(f"Plot saved as '{source_dir}/overall_scores_comparison.png'")


def main(args):
    source_dir = args.source_dir

    all_scores = []
    score_keys = set()

    for dir_name in os.listdir(source_dir):
        dir_path = os.path.join(source_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        mixeval_path = os.path.join(dir_path, args.benchmark)
        if not os.path.exists(mixeval_path):
            continue

        # Find the most recent date directory
        version_path = os.path.join(mixeval_path, args.benchmark_version)
        if not os.path.exists(version_path):
            continue

        for file in os.listdir(version_path):
            if file.startswith("score_") and file.endswith(".json"):
                file_path = os.path.join(version_path, file)
                judge_model = get_model_name(file)

                try:
                    scores = read_score_file(file_path)
                    scores["tested_model"] = dir_name
                    scores["judge_model"] = judge_model
                    all_scores.append(scores)
                    score_keys.update(scores.keys())
                except json.JSONDecodeError:
                    print(f"Error reading {file_path}. Skipping this file.")

    # Remove 'tested_model' and 'judge_model' from score_keys and sort them
    try:
        score_keys.remove("tested_model")
    except KeyError:
        pass
    try:
        score_keys.remove("judge_model")
    except KeyError:
        pass
    score_keys = sorted(score_keys)

    # Prepare the table data
    table_data = []
    for score in all_scores:
        row = [score["tested_model"], score["judge_model"]]
        for key in score_keys:
            row.append(score.get(key, ""))
        table_data.append(row)

    # Sort the table data by tested model and then judge model
    table_data.sort(key=lambda x: (x[0], x[1]))

    # Prepare headers
    headers = ["Tested Model", "Judge Model"] + score_keys

    # Print formatted table
    print("Formatted Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))

    # Generate and print CSV
    print("\nComma-Separated Values (CSV) for Google Sheets:")
    csv_output = generate_csv(headers, table_data)
    print(csv_output)

    # Plot overall results
    plot_overall_results(all_scores, source_dir)

    # Save to a file
    file_name = f"{source_dir}/output.csv"
    with open(file_name, "w", newline="") as f:
        f.write(csv_output)

    # Use prettytable to print the table
    table = PrettyTable()
    table.field_names = headers
    for row in table_data:
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        "-s",
        type=str,
        default="eval/chat_benchmarks/MixEval/results",
        help="Path to the source directory",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default="mixeval",
        help="Benchmark to use",
    )
    parser.add_argument(
        "--benchmark_version",
        "-v",
        type=str,
        default="2024-06-01",
        help="Benchmark version to use",
    )
    args = parser.parse_args()

    main(args)
