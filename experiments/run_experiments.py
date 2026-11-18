import sys
from pathlib import Path
import mesa
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cleaning_model"))

from model import CleaningModel

SCRIPT_LOCATION = Path(__file__).resolve().parent
ROOT = SCRIPT_LOCATION.parent
RESULTS_DIR = ROOT / "results"


COLORS = {
    "BLUE": "#2a78d6",
    "ORANGE": "#eb6834",
    "YELLOW": "#eda100",
    "PURPLE": "#b24e97",
    "RED": "#e70e0e",
}


def _log2_axis_scale(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def _ci95(values):
    """Half-width of the 95% confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    return stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)


def run_experiments(
    AGENT_CONFIGURATIONS=None, MAX_STEPS=50_000, DIRTY_PCT=0.5, N_SEEDS=30
):
    results = mesa.batch_run(
        CleaningModel,
        parameters={
            "width": 20,
            "height": 20,
            "dirty_pct": DIRTY_PCT,
            "max_steps": MAX_STEPS,
            "n_agents": AGENT_CONFIGURATIONS,
        },
        rng=range(N_SEEDS),
        data_collection_period=-1,
        max_steps=MAX_STEPS,
        number_processes=None,
    )

    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset="RunId").drop(columns=["AgentID", "Moves"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "sweep.csv", index=False)

    # One row per agent count. Everything the plots need is computed here,
    # so the plot functions only use summary_df.
    summary_df = (
        df.groupby("n_agents")
        .agg(
            AverageSteps=("Step", "mean"),
            StdSteps=("Step", "std"),
            CI95Steps=("Step", _ci95),
            MinSteps=("Step", "min"),
            Q1Steps=("Step", lambda x: x.quantile(0.25)),
            MedianSteps=("Step", "median"),
            Q3Steps=("Step", lambda x: x.quantile(0.75)),
            MaxSteps=("Step", "max"),
            AverageMoves=("TotalMoves", "mean"),
            CI95Moves=("TotalMoves", _ci95),
            AverageWallBumps=("WallBumps", "mean"),
            AverageBlockedMoves=("BlockedMoves", "mean"),
            AverageCleaningPercentage=("CleaningPercentage", "mean"),
            CI95CleaningPercentage=("CleaningPercentage", _ci95),
            SuccessfulRuns=("FullCleaningTime", "count"),
            Runs=("Step", "size"),
        )
        .reset_index()
    )
    summary_df["MovesPerAgent"] = summary_df["AverageMoves"] / summary_df["n_agents"]

    # Speedup T(1)/T(k) per seed: each seed is the same room for every agent count.
    times = df.pivot_table(index="seed", columns="n_agents", values="Step")
    per_seed_speedup = times.rdiv(times[times.columns.min()], axis=0)
    summary_df["Speedup"] = per_seed_speedup.mean().to_numpy()
    summary_df["CI95Speedup"] = per_seed_speedup.apply(_ci95).to_numpy()
    summary_df["Efficiency"] = summary_df["Speedup"] / summary_df["n_agents"]
    summary_df["CI95Efficiency"] = summary_df["CI95Speedup"] / summary_df["n_agents"]
    summary_df["MaxStepsLimit"] = MAX_STEPS
    summary_df["DirtyPct"] = DIRTY_PCT

    summary_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)

    # print_statistics(df, summary_df)

    return summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR


def plot_average_steps(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    plt.figure(figsize=(8, 5))

    plt.bar(summary_df["n_agents"], summary_df["AverageSteps"])

    plt.xlabel("Number of Robots")
    plt.ylabel("Average Steps")
    plt.title("Average Steps vs Number of Robots")
    plt.xticks(AGENT_CONFIGURATIONS)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if RESULTS_DIR:
        plt.savefig(RESULTS_DIR / "average_steps.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_average_movements(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    plt.figure(figsize=(8, 5))

    plt.bar(summary_df["n_agents"], summary_df["AverageMoves"])

    plt.xlabel("Number of Robots")
    plt.ylabel("Average Movements")
    plt.title("Average Movements vs Number of Robots")
    plt.xticks(AGENT_CONFIGURATIONS)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if RESULTS_DIR:
        plt.savefig(RESULTS_DIR / "average_movements.png")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_time_distribution(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    """Boxplots drawn from the quartiles stored in summary_df."""
    box_stats = [
        {
            "label": str(row.n_agents),
            "whislo": row.MinSteps,
            "q1": row.Q1Steps,
            "med": row.MedianSteps,
            "q3": row.Q3Steps,
            "whishi": row.MaxSteps,
            "mean": row.AverageSteps,
        }
        for row in summary_df.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bxp(
        box_stats,
        showfliers=False,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#cde2fb", "edgecolor": COLORS["BLUE"]},
        medianprops={"color": COLORS["BLUE"], "linewidth": 2},
        whiskerprops={"color": COLORS["BLUE"]},
        capprops={"color": COLORS["BLUE"]},
        meanprops={
            "marker": "D",
            "markerfacecolor": COLORS["ORANGE"],
            "markeredgecolor": "none",
        },
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Steps until clean (log scale)")
    ax.set_title(
        "Distribution of Cleaning Time (box = quartiles, whiskers = min/max, diamond = mean)"
    )
    # finish(fig, RESULTS_DIR, "time_distribution.png", show_plot)
    plt.show()


def plot_speedup(summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False):
    k = summary_df["n_agents"].to_numpy(dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(k, k, "--", color="#8a8a85", linewidth=1.5, label="Ideal speedup")
    ax1.errorbar(
        k,
        summary_df["Speedup"],
        yerr=summary_df["CI95Speedup"],
        fmt="o-",
        color=COLORS["BLUE"],
        capsize=3,
        label="Measured (mean, 95% CI)",
    )
    ax1.set_yscale("log", base=2)

    _log2_axis_scale(ax1, AGENT_CONFIGURATIONS)
    ax1.set_yticks(AGENT_CONFIGURATIONS)
    ax1.set_yticklabels([str(t) for t in AGENT_CONFIGURATIONS])
    ax1.set_xlabel("Number of Robots")
    ax1.set_ylabel("Speedup  T(1) / T(k)")
    ax1.set_title("Speedup")
    ax1.legend()

    ax2.axhline(1, linestyle="--", color="#8a8a85", linewidth=1.5)
    ax2.errorbar(
        k,
        summary_df["Efficiency"],
        yerr=summary_df["CI95Efficiency"],
        fmt="o-",
        color=COLORS["BLUE"],
        capsize=3,
    )

    _log2_axis_scale(ax2, AGENT_CONFIGURATIONS)

    upper = (summary_df["Efficiency"] + summary_df["CI95Efficiency"].fillna(0)).max()
    ax2.set_ylim(0, max(1.2, upper * 1.05))
    ax2.set_xlabel("Number of Robots")
    ax2.set_ylabel("Efficiency  speedup / k")
    ax2.set_title("Efficiency (1 = every robot fully useful)")
    # _finish(fig, RESULTS_DIR, "speedup_efficiency.png", show_plot)
    plt.show()


def plot_failed_moves(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        summary_df["n_agents"],
        summary_df["AverageWallBumps"],
        "o-",
        color=COLORS["BLUE"],
        label="Hit a wall",
    )
    ax.plot(
        summary_df["n_agents"],
        summary_df["AverageBlockedMoves"],
        "o-",
        color=COLORS["ORANGE"],
        label="Blocked by another robot",
    )
    _log2_axis_scale(ax, AGENT_CONFIGURATIONS)
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Failed moves per run (mean)")
    ax.set_title("Failed Move Attempts: Walls vs Congestion")
    ax.legend()
    # _finish(fig, RESULTS_DIR, "failed_moves.png", show_plot)
    plt.show()


def plot_cleaning_percentage_comparison(
    summaries, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    """Final clean percentage per robot count, one line per step limit."""
    colors = [
        COLORS["BLUE"],
        COLORS["ORANGE"],
        "#1baf7a",
        COLORS["YELLOW"],
        COLORS["PURPLE"],
        COLORS["RED"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, summary_df in zip(colors, summaries):
        limit = summary_df["MaxStepsLimit"].iloc[0]
        ax.errorbar(
            summary_df["n_agents"],
            summary_df["AverageCleaningPercentage"],
            yerr=summary_df["CI95CleaningPercentage"].fillna(0),
            fmt="o-",
            color=color,
            capsize=3,
            label=f"Limit of {limit:,} steps",
        )
    _log2_axis_scale(ax, AGENT_CONFIGURATIONS)

    ax.set_ylim(top=101)
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Clean cells at the end (%)")
    ax.set_title("Final Cleaning Percentage vs Number of Robots (mean, 95% CI)")
    ax.legend(loc="lower right")
    # _finish(fig, RESULTS_DIR, "cleaning_percentage_comparison.png", show_plot)
    plt.show()


def plot_cleaning_percentage_heatmap(
    summaries, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    """Final clean percentage for every step limit (rows) and robot count (columns)."""
    combined = pd.concat(summaries, ignore_index=True)
    table = combined.pivot_table(
        index="MaxStepsLimit", columns="n_agents", values="AverageCleaningPercentage"
    ).sort_index()

    # Single-hue sequential scale: light = less clean, dark = fully clean
    cmap = LinearSegmentedColormap.from_list(
        "clean",
        ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    )
    vmin, vmax = np.floor(table.values.min()), 100

    fig, ax = plt.subplots(
        figsize=(1.1 * len(table.columns) + 2.5, 0.6 * len(table.index) + 1.8)
    )
    image = ax.imshow(table.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            # Round down so 99.96 shows as 99.9, not as a misleading 100.0
            label = "100" if value >= 100 else f"{np.floor(value * 10) / 10:.1f}"
            is_dark = (value - vmin) / (vmax - vmin) > 0.55
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if is_dark else "#1f1f1e",
            )

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(k) for k in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{int(limit):,}" for limit in table.index])
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Step limit")
    ax.set_title("Clean cells at the end (%), mean over seeds")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(table.columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table.index)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2, alpha=1)
    ax.tick_params(which="both", length=0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Clean cells (%)")
    colorbar.outline.set_visible(False)
    plt.show()


def cleaning_percentage_table(summaries):
    combined = pd.concat(summaries, ignore_index=True)
    return combined.pivot_table(
        index="n_agents", columns="MaxStepsLimit", values="AverageCleaningPercentage"
    ).round(2)


def dirt_table(summaries, value="AverageSteps"):
    combined = pd.concat(summaries, ignore_index=True)
    table = combined.pivot_table(index="n_agents", columns="DirtyPct", values=value)
    table.columns = [f"{pct:.0%} dirty" for pct in table.columns]
    return table.round(1)


def plot_dirt_comparison(
    summaries, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    """Cleaning time and total movements per robot count, one line per dirt percentage."""
    summaries = sorted(summaries, key=lambda s: s["DirtyPct"].iloc[0])
    # Dirt percentage is ordered, so use one hue from light (little dirt) to dark (much dirt)
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    colors = (
        ramp[-len(summaries) :]
        if len(summaries) <= len(ramp)
        else plt.cm.Blues(np.linspace(0.35, 1, len(summaries)))
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for color, summary_df in zip(colors, summaries):
        label = f"{summary_df['DirtyPct'].iloc[0]:.0%} dirty"
        ax1.errorbar(
            summary_df["n_agents"],
            summary_df["AverageSteps"],
            yerr=summary_df["CI95Steps"],
            fmt="o-",
            color=color,
            capsize=3,
            label=label,
        )
        ax2.errorbar(
            summary_df["n_agents"],
            summary_df["AverageMoves"],
            yerr=summary_df["CI95Moves"],
            fmt="o-",
            color=color,
            capsize=3,
            label=label,
        )

    ax1.set_yscale("log")
    _log2_axis_scale(ax1, AGENT_CONFIGURATIONS)
    ax1.set_yticks([250, 500, 1_000, 2_500, 5_000])
    ax1.set_yticklabels(["250", "500", "1,000", "2,500", "5,000"])
    ax1.yaxis.set_minor_formatter(plt.NullFormatter())
    ax1.set_xlabel("Number of Robots")
    ax1.set_ylabel("Steps until clean (log scale)")
    ax1.set_title("Cleaning Time (mean, 95% CI)")

    _log2_axis_scale(ax2, AGENT_CONFIGURATIONS)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Number of Robots")
    ax2.set_ylabel("Total Movements (all robots)")
    ax2.set_title("Total Movements (mean, 95% CI)")
    ax2.legend(title="Initial dirt", loc="upper left", bbox_to_anchor=(1.02, 1))

    # _finish(fig, RESULTS_DIR, "dirt_comparison.png", show_plot)


if __name__ == "__main__":
    AGENT_CONFIGURATIONS = [1, 2, 4, 8]
    MAX_STEPS = 50_000
    df, agent_config, results_dir = run_experiments()
    plot_average_steps(df, agent_config, results_dir)
    plot_average_movements(df, agent_config, results_dir)
