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

ScriptLocation = Path(__file__).resolve().parent
root = ScriptLocation.parent
resultsDir = root / "results"


COLORS = {
	"BLUE": "#2a78d6",
	"ORANGE": "#eb6834",
	"YELLOW": "#eda100",
	"PURPLE": "#b24e97",
	"RED": "#e70e0e",
}


# Purpose: set the X axis to log base 2, with ticks at the given values and labels as strings.
# Parameters: ax, matplotlib axes to modify, ticks, list of values for the ticks.
# Returns: nothing.
def log2AxisScale(ax, ticks):
	ax.set_xscale("log", base=2)
	ax.set_xticks(ticks)
	ax.set_xticklabels([str(t) for t in ticks])
	ax.minorticks_off()


# Purpose: compute the half-width of the 95% confidence interval of the mean.
# Parameters: values, sequence of numeric values of a sample.
# Returns: half-width of the interval (float), or NaN if there are fewer than 2 values.
def ci95(values):
	values = np.asarray(values, dtype=float)
	if len(values) < 2:
		return np.nan
	return stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)


# Purpose: run the sweep (30 seeds per configuration) with mesa.batch_run.
# Parameters: agentConfigurations, list with the agent counts to test, maxSteps, maximum time of each run.
# Returns: DataFrame with one row per run (batch_run returns one row per agent, so the extras are dropped).
def runSweep(agentConfigurations, maxSteps, dirtyPct=0.5, nSeeds=30):
	results = mesa.batch_run(
		CleaningModel,
		parameters={
			"width": 20,
			"height": 20,
			"dirtyPct": dirtyPct,
			"maxSteps": maxSteps,
			"numAgents": agentConfigurations,
		},
		rng=range(nSeeds),
		data_collection_period=-1,
		max_steps=maxSteps,
		number_processes=None,
	)

	df = pd.DataFrame(results)
	return df.drop_duplicates(subset="RunId").drop(columns=["AgentID", "Moves"])


# Purpose: compute the statistics of the sweep for each agent count.
# Parameters: df, DataFrame with one row per run, as returned by runSweep.
# Returns: DataFrame with one row per agent count.
def buildSummary(df):
	# One row per agent count. Everything the plots need is computed here,
	# so the plot functions only use summaryDf.
	summaryDf = (
		df.groupby("numAgents")
		.agg(
			AverageSteps=("Step", "mean"),
			StdSteps=("Step", "std"),
			CI95Steps=("Step", ci95),
			MinSteps=("Step", "min"),
			Q1Steps=("Step", lambda x: x.quantile(0.25)),
			MedianSteps=("Step", "median"),
			Q3Steps=("Step", lambda x: x.quantile(0.75)),
			MaxSteps=("Step", "max"),
			AverageMoves=("TotalMoves", "mean"),
			CI95Moves=("TotalMoves", ci95),
			AverageWallBumps=("WallBumps", "mean"),
			AverageBlockedMoves=("BlockedMoves", "mean"),
			AverageCleaningPercentage=("CleaningPercentage", "mean"),
			CI95CleaningPercentage=("CleaningPercentage", ci95),
			SuccessfulRuns=("FullCleaningTime", "count"),
			Runs=("Step", "size"),
		)
		.reset_index()
	)
	summaryDf["MovesPerAgent"] = summaryDf["AverageMoves"] / summaryDf["numAgents"]
	return summaryDf


# Purpose: add the speedup T(1)/T(k) and the efficiency (speedup / k) with their confidence intervals.
# Parameters: summaryDf, summary table from buildSummary (not modified), df, DataFrame with one row per run.
# Returns: a copy of summaryDf with the Speedup, CI95Speedup, Efficiency and CI95Efficiency columns.
def addSpeedup(summaryDf, df):
	summaryDf = summaryDf.copy()

	# Speedup T(1)/T(k) per seed: each seed is the same room for every agent count.
	times = df.pivot_table(index="seed", columns="numAgents", values="Step")
	perSeedSpeedup = times.rdiv(times[times.columns.min()], axis=0)
	summaryDf["Speedup"] = perSeedSpeedup.mean().to_numpy()
	summaryDf["CI95Speedup"] = perSeedSpeedup.apply(ci95).to_numpy()
	summaryDf["Efficiency"] = summaryDf["Speedup"] / summaryDf["numAgents"]
	summaryDf["CI95Efficiency"] = summaryDf["CI95Speedup"] / summaryDf["numAgents"]
	return summaryDf


# Purpose: save the raw runs and the summary table as CSV files in resultsDir.
# Parameters: df, DataFrame with one row per run, summaryDf, summary table per agent count.
# Returns: nothing.
def saveResults(df, summaryDf):
	resultsDir.mkdir(parents=True, exist_ok=True)
	df.to_csv(resultsDir / "sweep.csv", index=False)
	summaryDf.to_csv(resultsDir / "experiment_summary.csv", index=False)


# Purpose: run the whole experiment: sweep, statistics, speedup and CSV files.
# Parameters: agentConfigurations, list with the agent counts to test, maxSteps, maximum time of each run.
# Returns: tuple (summary table per agent count, agentConfigurations, results folder).
def runExperiments(agentConfigurations=None, maxSteps=50_000, dirtyPct=0.5, nSeeds=30):
	df = runSweep(agentConfigurations, maxSteps, dirtyPct, nSeeds)
	summaryDf = addSpeedup(buildSummary(df), df)
	summaryDf["MaxStepsLimit"] = maxSteps
	summaryDf["DirtyPct"] = dirtyPct
	saveResults(df, summaryDf)

	return summaryDf, agentConfigurations, resultsDir


# Purpose: save a figure as PNG (if a folder is given), then show it or close it.
# Parameters: fig, matplotlib figure to finish, resultsDir, folder where the PNG is saved (None = do not save),
#             fileName, name of the PNG file, showPlot, True to show the figure, False to close it.
# Returns: nothing.
def finishFigure(fig, resultsDir, fileName, showPlot):
	if resultsDir:
		fig.savefig(resultsDir / fileName)

	if showPlot:
		plt.show()
	else:
		plt.close(fig)


# Purpose: bar chart of the average steps until clean versus the number of robots.
# Parameters: summaryDf, summary table; agentConfigurations, X axis values,
#             resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotAverageSteps(summaryDf, agentConfigurations, resultsDir=None, showPlot=False):
	fig = plt.figure(figsize=(8, 5))

	plt.bar(summaryDf["numAgents"], summaryDf["AverageSteps"])

	plt.xlabel("Number of Robots")
	plt.ylabel("Average Steps")
	plt.title("Average Steps vs Number of Robots")
	plt.xticks(agentConfigurations)
	plt.grid(axis="y", alpha=0.3)

	plt.tight_layout()
	finishFigure(fig, resultsDir, "average_steps.png", showPlot)


# Purpose: bar chart of the average total moves versus the number of robots.
# Parameters: summaryDf, summary table; agentConfigurations, X axis values,
#             resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotAverageMovements(
	summaryDf, agentConfigurations, resultsDir=None, showPlot=False
):
	fig = plt.figure(figsize=(8, 5))

	plt.bar(summaryDf["numAgents"], summaryDf["AverageMoves"])

	plt.xlabel("Number of Robots")
	plt.ylabel("Average Movements")
	plt.title("Average Movements vs Number of Robots")
	plt.xticks(agentConfigurations)
	plt.grid(axis="y", alpha=0.3)

	plt.tight_layout()
	finishFigure(fig, resultsDir, "average_movements.png", showPlot)


# Purpose: draw boxplots of the cleaning time from the quartiles stored in summaryDf.
# Parameters: summaryDf, summary table; agentConfigurations, unused here (kept so the call
#             matches the other plot functions), resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotTimeDistribution(summaryDf, resultsDir=None, showPlot=False):
	boxStats = [
		{
			"label": str(row.numAgents),
			"whislo": row.MinSteps,
			"q1": row.Q1Steps,
			"med": row.MedianSteps,
			"q3": row.Q3Steps,
			"whishi": row.MaxSteps,
			"mean": row.AverageSteps,
		}
		for row in summaryDf.itertuples()
	]
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.bxp(
		boxStats,
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
	finishFigure(fig, resultsDir, "time_distribution.png", showPlot)


# Purpose: draw the measured speedup T(1)/T(k) against the ideal speedup on one axes.
# Parameters: ax, matplotlib axes to draw on, summaryDf, summary table, agentConfigurations, axis values.
# Returns: nothing.
def drawSpeedup(ax, summaryDf, agentConfigurations):
	k = summaryDf["numAgents"].to_numpy(dtype=float)

	ax.plot(k, k, "--", color="#8a8a85", linewidth=1.5, label="Ideal speedup")
	ax.errorbar(
		k,
		summaryDf["Speedup"],
		yerr=summaryDf["CI95Speedup"],
		fmt="o-",
		color=COLORS["BLUE"],
		capsize=3,
		label="Measured (mean, 95% CI)",
	)
	ax.set_yscale("log", base=2)

	log2AxisScale(ax, agentConfigurations)
	ax.set_yticks(agentConfigurations)
	ax.set_yticklabels([str(t) for t in agentConfigurations])
	ax.set_xlabel("Number of Robots")
	ax.set_ylabel("Speedup  T(1) / T(k)")
	ax.set_title("Speedup")
	ax.legend()


# Purpose: draw the efficiency (speedup / k) on one axes, with 1 as the ideal reference line.
# Parameters: ax, matplotlib axes to draw on, summaryDf, summary table, agentConfigurations, axis values.
# Returns: nothing.
def drawEfficiency(ax, summaryDf, agentConfigurations):
	k = summaryDf["numAgents"].to_numpy(dtype=float)

	ax.axhline(1, linestyle="--", color="#8a8a85", linewidth=1.5)
	ax.errorbar(
		k,
		summaryDf["Efficiency"],
		yerr=summaryDf["CI95Efficiency"],
		fmt="o-",
		color=COLORS["BLUE"],
		capsize=3,
	)

	log2AxisScale(ax, agentConfigurations)
	upper = (summaryDf["Efficiency"] + summaryDf["CI95Efficiency"].fillna(0)).max()
	ax.set_ylim(0, max(1.2, upper * 1.05))
	ax.set_xlabel("Number of Robots")
	ax.set_ylabel("Efficiency  speedup / k")
	ax.set_title("Efficiency (1 = every robot fully useful)")


# Purpose: plot the speedup T(1)/T(k) and the efficiency (speedup / k) versus the number of robots.
# Parameters: summaryDf, summary table, agentConfigurations, axis values,
#             resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotSpeedup(summaryDf, agentConfigurations, resultsDir=None, showPlot=False):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

	drawSpeedup(ax1, summaryDf, agentConfigurations)
	drawEfficiency(ax2, summaryDf, agentConfigurations)

	finishFigure(fig, resultsDir, "speedup_efficiency.png", showPlot)


# Purpose: plot the failed move attempts (wall bumps and moves blocked by another robot).
# Parameters: summaryDf, summary table, agentConfigurations, X axis values,
#             resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotFailedMoves(summaryDf, agentConfigurations, resultsDir=None, showPlot=False):
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(
		summaryDf["numAgents"],
		summaryDf["AverageWallBumps"],
		"o-",
		color=COLORS["BLUE"],
		label="Hit a wall",
	)
	ax.plot(
		summaryDf["numAgents"],
		summaryDf["AverageBlockedMoves"],
		"o-",
		color=COLORS["ORANGE"],
		label="Blocked by another robot",
	)
	log2AxisScale(ax, agentConfigurations)
	ax.set_xlabel("Number of Robots")
	ax.set_ylabel("Failed moves per run (mean)")
	ax.set_title("Failed Move Attempts: Walls vs Congestion")
	ax.legend()
	finishFigure(fig, resultsDir, "failed_moves.png", showPlot)


# Purpose: plot the final % of clean cells per robot count, one line per step limit.
# Parameters: summaries, list of summary tables (one per step limit), agentConfigurations,
#             X axis values, resultsDir, folder where the PNG is saved (None = do not save),
#             showPlot, show the figure.
# Returns: nothing.
def plotCleaningPercentageComparison(
	summaries, agentConfigurations, resultsDir=None, showPlot=False
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
	for color, summaryDf in zip(colors, summaries):
		limit = summaryDf["MaxStepsLimit"].iloc[0]
		ax.errorbar(
			summaryDf["numAgents"],
			summaryDf["AverageCleaningPercentage"],
			yerr=summaryDf["CI95CleaningPercentage"].fillna(0),
			fmt="o-",
			color=color,
			capsize=3,
			label=f"Limit of {limit:,} steps",
		)
	log2AxisScale(ax, agentConfigurations)

	ax.set_ylim(top=101)
	ax.set_xlabel("Number of Robots")
	ax.set_ylabel("Clean cells at the end (%)")
	ax.set_title("Final Cleaning Percentage vs Number of Robots (mean, 95% CI)")
	ax.legend(loc="lower right")
	finishFigure(fig, resultsDir, "cleaning_percentage_comparison.png", showPlot)


# Purpose: build the table behind the heatmap: average final % of clean cells per step limit and robot count.
# Parameters: summaries, list of summary tables (one per step limit).
# Returns: DataFrame with one row per step limit (sorted) and one column per robot count.
def buildHeatmapTable(summaries):
	combined = pd.concat(summaries, ignore_index=True)
	return combined.pivot_table(
		index="MaxStepsLimit", columns="numAgents", values="AverageCleaningPercentage"
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
			isDark = (value - vmin) / (vmax - vmin) > 0.55
			ax.text(
				j,
				i,
				label,
				ha="center",
				va="center",
				fontsize=9,
				color="white" if isDark else "#1f1f1e",
			)


# Purpose: draw a heatmap of the final % of clean cells per step limit (rows) and robot count (columns).
# Parameters: summaries, list of summary tables (one per step limit), agentConfigurations,
#             unused here (kept so the call matches the other plot functions),
#             resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotCleaningPercentageHeatmap(
	summaries, agentConfigurations, resultsDir=None, showPlot=False
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
	finishFigure(fig, resultsDir, "cleaning_percentage_heatmap.png", showPlot)


# Purpose: build the table behind the cleaning percentage comparison plot: average final % of clean cells per robot count, one row per step limit.
# Parameters: summaries, list of summary tables (one per step limit).
# Returns: DataFrame with one row per step limit (sorted) and one column per robot count.
def cleaningPercentageTable(summaries):
	combined = pd.concat(summaries, ignore_index=True)
	return combined.pivot_table(
		index="numAgents", columns="MaxStepsLimit", values="AverageCleaningPercentage"
	).round(2)


# Purpose: build the table behind the dirt comparison plot: average steps and total moves per robot count, one row per dirt percentage.
# Parameters: summaries, list of summary tables (one per dirt percentage).
# Returns: DataFrame with one row per dirt percentage (sorted) and one column per robot count.
def dirtTable(summaries, value="AverageSteps"):
	combined = pd.concat(summaries, ignore_index=True)
	table = combined.pivot_table(index="numAgents", columns="DirtyPct", values=value)
	table.columns = [f"{pct:.0%} dirty" for pct in table.columns]
	return table.round(1)


# Purpose: plot the cleaning time and total movements per robot count, one line per dirt percentage.
# Parameters: summaries, list of summary tables (one per dirt percentage), agentConfigurations, X axis values,
#             resultsDir, folder where the PNG is saved (None = do not save), showPlot, show the figure.
# Returns: nothing.
def plotDirtComparison(summaries, agentConfigurations, resultsDir=None, showPlot=False):
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
	for color, summary in zip(colors, summaries):
		label = f"{summary['DirtyPct'].iloc[0]:.0%} dirty"
		ax1.errorbar(
			summary["numAgents"],
			summary["AverageSteps"],
			yerr=summary["CI95Steps"],
			fmt="o-",
			color=color,
			capsize=3,
			label=label,
		)
		ax2.errorbar(
			summary["numAgents"],
			summary["AverageMoves"],
			yerr=summary["CI95Moves"],
			fmt="o-",
			color=color,
			capsize=3,
			label=label,
		)

	ax1.set_yscale("log")
	log2AxisScale(ax1, agentConfigurations)
	ax1.set_yticks([250, 500, 1_000, 2_500, 5_000])
	ax1.set_yticklabels(["250", "500", "1,000", "2,500", "5,000"])
	ax1.yaxis.set_minor_formatter(plt.NullFormatter())
	ax1.set_xlabel("Number of Robots")
	ax1.set_ylabel("Steps until clean (log scale)")
	ax1.set_title("Cleaning Time (mean, 95% CI)")

	log2AxisScale(ax2, agentConfigurations)
	ax2.set_ylim(bottom=0)
	ax2.set_xlabel("Number of Robots")
	ax2.set_ylabel("Total Movements (all robots)")
	ax2.set_title("Total Movements (mean, 95% CI)")
	ax2.legend(title="Initial dirt", loc="upper left", bbox_to_anchor=(1.02, 1))

	finishFigure(fig, resultsDir, "dirt_comparison.png", showPlot)


if __name__ == "__main__":
	agentConfigurations = [1, 2, 4, 8]
	maxSteps = 50_000
	summaryDf, agentConfig, resultsDir = runExperiments(agentConfigurations, maxSteps)
	plotAverageSteps(summaryDf, agentConfig, resultsDir)
	plotAverageMovements(summaryDf, agentConfig, resultsDir)
