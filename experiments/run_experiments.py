# Description: experiment sweep with mesa.batch_run (varying the number of agents), statistics per configuration and plots.
# Authors: 
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-17

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


# Purpose: compute the half-width of the 95% confidence interval of the mean.
# Parameters: values, sequence of numeric values of a sample.
# Returns: half-width of the interval (float), or NaN if there are fewer than 2 values.
def _ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    return stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)


# Purpose: run the sweep (30 seeds per configuration) with mesa.batch_run.
# Parameters: AGENT_CONFIGURATIONS, list with the agent counts to test, MAX_STEPS, maximum time of each run.
# Returns: DataFrame with one row per run (batch_run returns one row per agent, so the extras are dropped).
def runSweep(AGENT_CONFIGURATIONS, MAX_STEPS):
    results = mesa.batch_run(
        CleaningModel,
        parameters={
            "width": 20,
            "height": 20,
            "dirty_pct": 0.5,
            "max_steps": MAX_STEPS,
            "n_agents": AGENT_CONFIGURATIONS,
        },
        rng=range(30),
        data_collection_period=-1,
        max_steps=MAX_STEPS,
        number_processes=None,
    )

    df = pd.DataFrame(results)
    return df.drop_duplicates(subset="RunId").drop(columns=["AgentID", "Moves"])


# Purpose: compute the statistics of the sweep for each agent count.
# Parameters: df, DataFrame with one row per run, as returned by runSweep.
# Returns: DataFrame with one row per agent count.
def buildSummary(df):
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
    return summary_df


# Purpose: add the speedup T(1)/T(k) and the efficiency (speedup / k) with their confidence intervals.
# Parameters: summary_df, summary table from buildSummary (not modified), df, DataFrame with one row per run.
# Returns: a copy of summary_df with the Speedup, CI95Speedup, Efficiency and CI95Efficiency columns.
def addSpeedup(summary_df, df):
    summary_df = summary_df.copy()

    # Speedup T(1)/T(k) per seed: each seed is the same room for every agent count.
    times = df.pivot_table(index="seed", columns="n_agents", values="Step")
    per_seed_speedup = times.rdiv(times[times.columns.min()], axis=0)
    summary_df["Speedup"] = per_seed_speedup.mean().to_numpy()
    summary_df["CI95Speedup"] = per_seed_speedup.apply(_ci95).to_numpy()
    summary_df["Efficiency"] = summary_df["Speedup"] / summary_df["n_agents"]
    summary_df["CI95Efficiency"] = summary_df["CI95Speedup"] / summary_df["n_agents"]
    return summary_df


# Purpose: save the raw runs and the summary table as CSV files in RESULTS_DIR.
# Parameters: df, DataFrame with one row per run, summary_df, summary table per agent count.
# Returns: nothing.
def saveResults(df, summary_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "sweep.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)


# Purpose: run the whole experiment: sweep, statistics, speedup and CSV files.
# Parameters: AGENT_CONFIGURATIONS, list with the agent counts to test, MAX_STEPS, maximum time of each run.
# Returns: tuple (summary table per agent count, AGENT_CONFIGURATIONS, results folder).
def run_experiments(AGENT_CONFIGURATIONS=None, MAX_STEPS=50_000):
    df = runSweep(AGENT_CONFIGURATIONS, MAX_STEPS)
    summary_df = addSpeedup(buildSummary(df), df)
    summary_df["MaxStepsLimit"] = MAX_STEPS
    saveResults(df, summary_df)

    return summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR


# Purpose: save a figure as PNG (if a folder is given), then show it or close it.
# Parameters: fig, matplotlib figure to finish, RESULTS_DIR, folder where the PNG is saved (None = do not save),
#             fileName, name of the PNG file, show_plot, True to show the figure, False to close it.
# Returns: nothing.
def finish_figure(fig, RESULTS_DIR, fileName, show_plot):
    if RESULTS_DIR:
        fig.savefig(RESULTS_DIR / fileName)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# Purpose: bar chart of the average steps until clean versus the number of robots.
# Parameters: summary_df, summary table; AGENT_CONFIGURATIONS, X axis values,
#             RESULTS_DIR, folder where the PNG is saved (None = do not save), show_plot, show the figure.
# Returns: nothing.
def plot_average_steps(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    fig = plt.figure(figsize=(8, 5))

    plt.bar(summary_df["n_agents"], summary_df["AverageSteps"])

    plt.xlabel("Number of Robots")
    plt.ylabel("Average Steps")
    plt.title("Average Steps vs Number of Robots")
    plt.xticks(AGENT_CONFIGURATIONS)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    finish_figure(fig, RESULTS_DIR, "average_steps.png", show_plot)


# Purpose: bar chart of the average total moves versus the number of robots.
# Parameters: summary_df, summary table; AGENT_CONFIGURATIONS, X axis values,
#             RESULTS_DIR, folder where the PNG is saved (None = do not save), show_plot, show the figure.
# Returns: nothing.
def plot_average_movements(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    fig = plt.figure(figsize=(8, 5))

    plt.bar(summary_df["n_agents"], summary_df["AverageMoves"])

    plt.xlabel("Number of Robots")
    plt.ylabel("Average Movements")
    plt.title("Average Movements vs Number of Robots")
    plt.xticks(AGENT_CONFIGURATIONS)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    finish_figure(fig, RESULTS_DIR, "average_movements.png", show_plot)


# Purpose: draw boxplots of the cleaning time from the quartiles stored in summary_df.
# Parameters: summary_df, summary table; AGENT_CONFIGURATIONS, unused here (kept so the call
#             matches the other plot functions), RESULTS_DIR, folder where the PNG is saved (None = do not save), show_plot, show the figure.
# Returns: nothing.
def plot_time_distribution(
    summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
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
    finish_figure(fig, RESULTS_DIR, "time_distribution.png", show_plot)


# Purpose: draw the measured speedup T(1)/T(k) against the ideal speedup on one axes.
# Parameters: ax, matplotlib axes to draw on, summary_df, summary table, AGENT_CONFIGURATIONS, axis values.
# Returns: nothing.
def drawSpeedup(ax, summary_df, AGENT_CONFIGURATIONS):
    k = summary_df["n_agents"].to_numpy(dtype=float)

    ax.plot(k, k, "--", color="#8a8a85", linewidth=1.5, label="Ideal speedup")
    ax.errorbar(
        k,
        summary_df["Speedup"],
        yerr=summary_df["CI95Speedup"],
        fmt="o-",
        color=COLORS["BLUE"],
        capsize=3,
        label="Measured (mean, 95% CI)",
    )
    ax.set_yscale("log", base=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(AGENT_CONFIGURATIONS)
    ax.set_xticklabels([str(t) for t in AGENT_CONFIGURATIONS])
    ax.minorticks_off()
    ax.set_yticks(AGENT_CONFIGURATIONS)
    ax.set_yticklabels([str(t) for t in AGENT_CONFIGURATIONS])
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Speedup  T(1) / T(k)")
    ax.set_title("Speedup")
    ax.legend()


# Purpose: draw the efficiency (speedup / k) on one axes, with 1 as the ideal reference line.
# Parameters: ax, matplotlib axes to draw on, summary_df, summary table, AGENT_CONFIGURATIONS, axis values.
# Returns: nothing.
def drawEfficiency(ax, summary_df, AGENT_CONFIGURATIONS):
    k = summary_df["n_agents"].to_numpy(dtype=float)

    ax.axhline(1, linestyle="--", color="#8a8a85", linewidth=1.5)
    ax.errorbar(
        k,
        summary_df["Efficiency"],
        yerr=summary_df["CI95Efficiency"],
        fmt="o-",
        color=COLORS["BLUE"],
        capsize=3,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(AGENT_CONFIGURATIONS)
    ax.set_xticklabels([str(t) for t in AGENT_CONFIGURATIONS])
    ax.minorticks_off()

    upper = (summary_df["Efficiency"] + summary_df["CI95Efficiency"].fillna(0)).max()
    ax.set_ylim(0, max(1.2, upper * 1.05))
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Efficiency  speedup / k")
    ax.set_title("Efficiency (1 = every robot fully useful)")


# Purpose: plot the speedup T(1)/T(k) and the efficiency (speedup / k) versus the number of robots.
# Parameters: summary_df, summary table, AGENT_CONFIGURATIONS, axis values,
#             RESULTS_DIR, folder where the PNG is saved (None = do not save), show_plot, show the figure.
# Returns: nothing.
def plot_speedup(summary_df, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    drawSpeedup(ax1, summary_df, AGENT_CONFIGURATIONS)
    drawEfficiency(ax2, summary_df, AGENT_CONFIGURATIONS)

    finish_figure(fig, RESULTS_DIR, "speedup_efficiency.png", show_plot)


# Purpose: plot the failed move attempts (wall bumps and moves blocked by another robot).
# Parameters: summary_df, summary table, AGENT_CONFIGURATIONS, X axis values,
#             RESULTS_DIR, folder where the PNG is saved (None = do not save), show_plot, show the figure.
# Returns: nothing.
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
    ax.set_xscale("log", base=2)
    ax.set_xticks(AGENT_CONFIGURATIONS)
    ax.set_xticklabels([str(t) for t in AGENT_CONFIGURATIONS])
    ax.minorticks_off()
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Failed moves per run (mean)")
    ax.set_title("Failed Move Attempts: Walls vs Congestion")
    ax.legend()
    finish_figure(fig, RESULTS_DIR, "failed_moves.png", show_plot)


# Purpose: plot the final % of clean cells per robot count, one line per step limit.
# Parameters: summaries, list of summary tables (one per step limit), AGENT_CONFIGURATIONS,
#             X axis values, RESULTS_DIR, folder where the PNG is saved (None = do not save),
#             show_plot, show the figure.
# Returns: nothing.
def plot_cleaning_percentage_comparison(
    summaries, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
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
    ax.set_xscale("log", base=2)
    ax.set_xticks(AGENT_CONFIGURATIONS)
    ax.set_xticklabels([str(t) for t in AGENT_CONFIGURATIONS])
    ax.minorticks_off()

    ax.set_ylim(top=101)
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Clean cells at the end (%)")
    ax.set_title("Final Cleaning Percentage vs Number of Robots (mean, 95% CI)")
    ax.legend(loc="lower right")
    finish_figure(fig, RESULTS_DIR, "cleaning_percentage_comparison.png", show_plot)


# Purpose: build the table behind the heatmap: average final % of clean cells per step limit and robot count.
# Parameters: summaries, list of summary tables (one per step limit).
# Returns: DataFrame with one row per step limit (sorted) and one column per robot count.
def buildHeatmapTable(summaries):
    combined = pd.concat(summaries, ignore_index=True)
    return combined.pivot_table(
        index="MaxStepsLimit", columns="n_agents", values="AverageCleaningPercentage"
    ).sort_index()


# Purpose: write the value of every cell of the table on top of the heatmap.
# Parameters: ax, matplotlib axes holding the heatmap, table, DataFrame from buildHeatmapTable,
#             vmin, vmax, lower and upper limits of the color scale (used to pick a readable text color).
# Returns: nothing.
def annotateHeatmap(ax, table, vmin, vmax):
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


# Purpose: draw a heatmap of the final % of clean cells per step limit (rows) and robot count (columns).
# Parameters: summaries, list of summary tables (one per step limit), AGENT_CONFIGURATIONS,
#             unused here (kept so the call matches the other plot functions),
#             RESULTS_DIR, folder where the PNG is saved (None = do not save), show_plot, show the figure.
# Returns: nothing.
def plot_cleaning_percentage_heatmap(
    summaries, AGENT_CONFIGURATIONS, RESULTS_DIR=None, show_plot=False
):
    table = buildHeatmapTable(summaries)

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

    annotateHeatmap(ax, table, vmin, vmax)

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
    finish_figure(fig, RESULTS_DIR, "cleaning_percentage_heatmap.png", show_plot)


if __name__ == "__main__":
    AGENT_CONFIGURATIONS = [1, 2, 4, 8]
    MAX_STEPS = 50_000
    df, agent_config, results_dir = run_experiments(AGENT_CONFIGURATIONS, MAX_STEPS)
    plot_average_steps(df, agent_config, results_dir)
    plot_average_movements(df, agent_config, results_dir)
