"""
SHARDS=N && cd $EVALCHEMY && source /leonardo_work/EUHPC_E03_068/DCFT_shared/mamba/bin/activate /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy && python eval/distributed/launch.py --model_name open-thoughts/OpenThinker-7B --tasks LiveCodeBench,AIME24,AIME25,AMC23,GPQADiamond,MATH500 --num_shards $SHARDS --watchdog
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# Function to convert time strings to minutes
def time_to_minutes(time_str):
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours * 60 + minutes + seconds / 60


# Data from the benchmarks - Leonardo
leonardo_shards = [1, 2, 4, 8, 16, 32, 64, 128]
leonardo_min_times = ["10:14:50", "03:08:48", "01:00:18", "00:25:12", "00:16:48", "00:13:21", "00:11:45", "00:08:04"]
leonardo_max_times = ["10:14:50", "06:56:54", "03:40:57", "01:50:33", "01:03:31", "00:37:13", "00:23:15", "00:19:01"]
leonardo_mean_times = ["10:14:50", "05:02:51", "02:36:41", "01:19:29", "00:43:05", "00:25:59", "00:18:39", "00:15:13"]
leonardo_gpu_hours = [10.1, 10.1, 10.4, 10.6, 11.5, 13.9, 19.9, 32.5]

# Data from the benchmarks - Capella
capella_shards = [1, 2, 4, 8, 16, 32, 64, 128]
capella_min_times = ["05:18:37", "01:34:58", "00:33:22", "00:16:56", "00:11:31", "00:08:35", "00:04:32", "00:03:12"]
capella_max_times = ["05:18:37", "03:51:51", "02:01:38", "01:03:20", "00:37:29", "00:20:55", "00:15:07", "00:14:54"]
capella_mean_times = ["05:18:37", "02:43:24", "01:25:02", "00:44:00", "00:25:37", "00:15:19", "00:11:07", "00:09:48"]
capella_gpu_hours = [5.3, 5.4, 5.7, 5.9, 6.8, 8.2, 11.9, 20.9]

# Convert times to minutes - Leonardo
leonardo_min_times_min = [time_to_minutes(t) for t in leonardo_min_times]
leonardo_max_times_min = [time_to_minutes(t) for t in leonardo_max_times]
leonardo_mean_times_min = [time_to_minutes(t) for t in leonardo_mean_times]

# Convert max times to hours for Pareto plot - Leonardo
leonardo_max_times_hours = [t / 60 for t in leonardo_max_times_min]

# Convert times to minutes - Capella
capella_min_times_min = [time_to_minutes(t) for t in capella_min_times]
capella_max_times_min = [time_to_minutes(t) for t in capella_max_times]
capella_mean_times_min = [time_to_minutes(t) for t in capella_mean_times]

# Convert max times to hours for Pareto plot - Capella
capella_max_times_hours = [t / 60 for t in capella_max_times_min]


# Create a formatter for the y-axis to display time in hours format (whole numbers only)
def format_time(x, pos):
    hours = x / 60  # Convert minutes to hours
    return f"{int(hours)}"  # Convert to integer to remove decimal places


time_formatter = FuncFormatter(format_time)

# Set up the figure with academic style
plt.figure(figsize=(10, 8))
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


# Create plots for Leonardo
def create_leonardo_plots():
    # Set up subplot layout with 3 plots
    fig = plt.figure(figsize=(10, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot 1: Execution Times
    ax1.plot(
        leonardo_shards, leonardo_min_times_min, "o-", color="#1f77b4", label="Min Time", linewidth=2, markersize=6
    )
    ax1.plot(
        leonardo_shards, leonardo_max_times_min, "o-", color="#d62728", label="Max Time", linewidth=2, markersize=6
    )
    ax1.plot(
        leonardo_shards, leonardo_mean_times_min, "o-", color="#2ca02c", label="Mean Time", linewidth=2, markersize=6
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(leonardo_shards)
    ax1.set_xticklabels(leonardo_shards)
    ax1.set_ylabel("Execution Time in Hours")
    ax1.set_xlabel("Number of Shards (log₂ scale)")
    ax1.set_title("Leonardo: Shard Execution Time vs. Number of Shards")
    ax1.yaxis.set_major_formatter(time_formatter)
    # Set y-axis ticks at whole number hours
    max_hours = max(leonardo_max_times_min) / 60
    ax1.set_yticks(np.arange(0, max_hours + 1, 1) * 60)  # Convert hours to minutes for the ticks
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(frameon=True, loc="upper right")

    # Add annotations for inflection points
    ax1.annotate(
        "Diminishing returns\non time reduction",
        xy=(64, time_to_minutes("00:18:39")),
        xytext=(64, time_to_minutes("01:00:00")),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Plot 2: GPU Hours
    ax2.plot(leonardo_shards, leonardo_gpu_hours, "o-", color="#ff7f0e", linewidth=2, markersize=6)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(leonardo_shards)
    ax2.set_xticklabels(leonardo_shards)
    ax2.set_ylabel("Total GPU Hours")
    ax2.set_xlabel("Number of Shards (log₂ scale)")
    ax2.set_title("Leonardo: Total GPU Hours vs. Number of Shards")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add annotation for optimal point
    ax2.annotate(
        "Increasing initialization\n and underutilization",
        xy=(8, 10.6),
        xytext=(8, 15),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Plot 3: GPU Hours vs Wall Clock Hours
    ax3.plot(leonardo_max_times_hours, leonardo_gpu_hours, "o-", color="#9467bd", linewidth=2, markersize=8)

    # Add shard number annotations to each point
    for i, (x, y, label) in enumerate(zip(leonardo_max_times_hours, leonardo_gpu_hours, leonardo_shards)):
        ax3.annotate(
            label,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )

    ax3.set_xlabel("Max Shard Execution Time in Hours")
    ax3.set_ylabel("Total GPU Hours")
    ax3.set_title("Leonardo: Resource Usage vs. Wall Clock")
    # Set y-axis limits to reduce blank space
    ax3.set_ylim(8, 35)
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Add text explaining the benchmark
    benchmark_text = (
        "Benchmark Details:\n"
        "GPU: 1x A100 64GB per shard (Leonardo CINECA)\n"
        "Model: open-thoughts/OpenThinker-7B\n"
        "Tasks: LiveCodeBench, AIME24, AIME25, AMC23, GPQADiamond, MATH500 (3,127 instances)"
    )
    fig.text(0.13, 0.01, benchmark_text, fontsize=10, ha="left")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("eval/distributed/benchmarking_leonardo.png", dpi=300, bbox_inches="tight")
    plt.close()


# Create plots for Capella
def create_capella_plots():
    # Set up subplot layout with 3 plots
    fig = plt.figure(figsize=(10, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot 1: Execution Times
    ax1.plot(capella_shards, capella_min_times_min, "o-", color="#1f77b4", label="Min Time", linewidth=2, markersize=6)
    ax1.plot(capella_shards, capella_max_times_min, "o-", color="#d62728", label="Max Time", linewidth=2, markersize=6)
    ax1.plot(
        capella_shards, capella_mean_times_min, "o-", color="#2ca02c", label="Mean Time", linewidth=2, markersize=6
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(capella_shards)
    ax1.set_xticklabels(capella_shards)
    ax1.set_ylabel("Execution Time in Hours")
    ax1.set_xlabel("Number of Shards (log₂ scale)")
    ax1.set_title("Capella: Shard Execution Time vs. Number of Shards")
    ax1.yaxis.set_major_formatter(time_formatter)
    # Set y-axis ticks at whole number hours
    max_hours = max(capella_max_times_min) / 60
    ax1.set_yticks(np.arange(0, max_hours + 1, 1) * 60)  # Convert hours to minutes for the ticks
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(frameon=True, loc="upper right")

    # Add annotations for inflection points
    ax1.annotate(
        "Diminishing returns\non time reduction",
        xy=(64, time_to_minutes("00:11:07")),
        xytext=(64, time_to_minutes("00:30:00")),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Plot 2: GPU Hours
    ax2.plot(capella_shards, capella_gpu_hours, "o-", color="#ff7f0e", linewidth=2, markersize=6)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(capella_shards)
    ax2.set_xticklabels(capella_shards)
    ax2.set_ylabel("Total GPU Hours")
    ax2.set_xlabel("Number of Shards (log₂ scale)")
    ax2.set_title("Capella: Total GPU Hours vs. Number of Shards")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add annotation for optimal point
    ax2.annotate(
        "Increasing initialization\n and underutilization",
        xy=(8, 5.9),
        xytext=(8, 12),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Plot 3: GPU Hours vs Wall Clock Hours
    ax3.plot(capella_max_times_hours, capella_gpu_hours, "o-", color="#9467bd", linewidth=2, markersize=8)

    # Add shard number annotations to each point
    for i, (x, y, label) in enumerate(zip(capella_max_times_hours, capella_gpu_hours, capella_shards)):
        ax3.annotate(
            label,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )

    ax3.set_xlabel("Max Shard Execution Time in Hours")
    ax3.set_ylabel("Total GPU Hours")
    ax3.set_title("Capella: Resource Usage vs. Wall Clock")
    # Set y-axis limits to reduce blank space
    ax3.set_ylim(4, 22)
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Add text explaining the benchmark
    benchmark_text = (
        "Benchmark Details:\n"
        "GPU: 1x H100 80GB per shard (Capella)\n"
        "Model: open-thoughts/OpenThinker-7B\n"
        "Tasks: LiveCodeBench, AIME24, AIME25, AMC23, GPQADiamond, MATH500 (3,127 instances)"
    )
    fig.text(0.13, 0.01, benchmark_text, fontsize=10, ha="left")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("eval/distributed/benchmarking_capella.png", dpi=300, bbox_inches="tight")
    plt.close()


# Create comparison plot
def create_comparison_plot():
    plt.figure(figsize=(10, 8))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # Plot GPU Hours vs Wall Clock Hours for both clusters
    plt.plot(
        leonardo_max_times_hours,
        leonardo_gpu_hours,
        "o-",
        color="#1f77b4",
        label="Leonardo (A100 64GB)",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        capella_max_times_hours,
        capella_gpu_hours,
        "o-",
        color="#ff7f0e",
        label="Capella (H100 80GB)",
        linewidth=2,
        markersize=8,
    )

    # Add shard number annotations to each point
    for i, (x, y, label) in enumerate(zip(leonardo_max_times_hours, leonardo_gpu_hours, leonardo_shards)):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )

    for i, (x, y, label) in enumerate(zip(capella_max_times_hours, capella_gpu_hours, capella_shards)):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )

    plt.xlabel("Max Shard Execution Time in Hours")
    plt.ylabel("Total GPU Hours")
    plt.title("Resource Usage vs. Wall Clock Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(frameon=True, loc="upper right")

    # Add text explaining the benchmark
    benchmark_text = (
        "Benchmark Details:\n"
        "Model: open-thoughts/OpenThinker-7B\n"
        "Tasks: LiveCodeBench, AIME24, AIME25, AMC23, GPQADiamond, MATH500 (3,127 instances)"
    )
    plt.figtext(0.13, 0.01, benchmark_text, fontsize=10, ha="left")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("eval/distributed/benchmarking_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# Create all plots
create_leonardo_plots()
create_capella_plots()
create_comparison_plot()
