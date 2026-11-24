# Description: Activity 2 experiments. Success probability of Max-Plus coordinated robots versus the random (reactive) robots of M1 on a 20 x 30 room, results table, plots and a simulation snapshot.
# Authors:
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-24

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ScriptLocation = Path(__file__).resolve().parent
sys.path.insert(0, str(ScriptLocation))
sys.path.insert(0, str(ScriptLocation.parent / "cleaning_model"))

from run_experiments import COLORS, ci95, finishFigure, resultsDir, runSweep
from model import CleaningModel
from coordination import buildEdges

WIDTH = 20
HEIGHT = 30
MAX_STEPS = 500
N_SEEDS = 100
AGENT_COUNTS = [1, 2, 3, 4, 5, 8, 16]
DIRTY_COUNTS = [100, 200]
MODES = ("reactive", "maxplus")
MODE_LABELS = {"reactive": "Random (M1)", "maxplus": "Max-Plus"}
MODE_COLORS = {"reactive": COLORS["ORANGE"], "maxplus": COLORS["BLUE"]}

SNAPSHOT_AGENTS = 4
SNAPSHOT_DIRTY = 100
SNAPSHOT_STEPS = (0, 25, 50)


# Purpose: run every combination of coordination mode and amount of dirt with the same seeds.
# Parameters: agentCounts, list with the robot counts to test, dirtyCounts, list with the initial number of dirty cells,
#             nSeeds, number of seeds (simulations) per configuration, maxSteps, maximum time of each run.
# Returns: DataFrame with one row per simulation, with the extra column InitialDirty.
def runActivity2Sweep(
	agentCounts=AGENT_COUNTS,
	dirtyCounts=DIRTY_COUNTS,
	nSeeds=N_SEEDS,
	maxSteps=MAX_STEPS,
):
	frames = []
	for mode in MODES:
		for dirtyCount in dirtyCounts:
			df = runSweep(
				agentCounts,
				maxSteps,
				dirtyPct=dirtyCount / (WIDTH * HEIGHT),
				nSeeds=nSeeds,
				coordination=mode,
				width=WIDTH,
				height=HEIGHT,
			)
			df["InitialDirty"] = dirtyCount
			frames.append(df)
	return pd.concat(frames, ignore_index=True)


# Purpose: build the results table, one row per (initial dirt, coordination mode, number of robots).
#          A run is a success if every dirty cell was cleaned within the step limit. The time of a failed run is the step limit.
# Parameters: df, DataFrame with one row per simulation, as returned by runActivity2Sweep.
# Returns: DataFrame with the success probability and the mean and std of time, clean percentage and total moves.
def buildActivity2Summary(df):
	df = df.assign(Success=df["FullCleaningTime"].notna())
	summaryDf = (
		df.groupby(["InitialDirty", "coordination", "numAgents"])
		.agg(
			SuccessProbability=("Success", "mean"),
			MeanSteps=("Step", "mean"),
			StdSteps=("Step", "std"),
			CI95Steps=("Step", ci95),
			MeanCleaningPercentage=("CleaningPercentage", "mean"),
			StdCleaningPercentage=("CleaningPercentage", "std"),
			MeanMoves=("TotalMoves", "mean"),
			StdMoves=("TotalMoves", "std"),
			Runs=("Step", "size"),
		)
		.reset_index()
	)
	return summaryDf


# Purpose: save the raw simulations and the results table as CSV files in resultsDir.
# Parameters: df, DataFrame with one row per simulation, summaryDf, results table.
# Returns: nothing.
def saveActivity2Results(df, summaryDf):
	resultsDir.mkdir(parents=True, exist_ok=True)
	df.to_csv(resultsDir / "activity2_sweep.csv", index=False)
	summaryDf.to_csv(resultsDir / "activity2_summary.csv", index=False)


# Purpose: build the results table for the report, with "mean ± std" text columns.
# Parameters: summaryDf, results table from buildActivity2Summary.
# Returns: DataFrame with one row per (initial dirt, method, number of robots).
def buildActivity2Table(summaryDf):
	return pd.DataFrame(
		{
			"Dirty": summaryDf["InitialDirty"],
			"Method": summaryDf["coordination"].map(MODE_LABELS),
			"Robots": summaryDf["numAgents"],
			"Success": summaryDf["SuccessProbability"].map("{:.2f}".format),
			"Steps": [
				f"{m:.1f} ± {s:.1f}"
				for m, s in zip(summaryDf["MeanSteps"], summaryDf["StdSteps"])
			],
			"Clean %": [
				f"{m:.1f} ± {s:.1f}"
				for m, s in zip(
					summaryDf["MeanCleaningPercentage"],
					summaryDf["StdCleaningPercentage"],
				)
			],
			"Moves": [
				f"{m:.0f} ± {s:.0f}"
				for m, s in zip(summaryDf["MeanMoves"], summaryDf["StdMoves"])
			],
		}
	)


# Purpose: print the results table with "mean ± std" columns.
# Parameters: summaryDf, results table from buildActivity2Summary.
# Returns: nothing.
def printActivity2Table(summaryDf):
	print(buildActivity2Table(summaryDf).to_string(index=False))


# Purpose: grouped bar chart of the success probability versus the number of robots, one bar per coordination mode.
# Parameters: summaryDf, results table, dirtyCount, initial number of dirty cells to plot,
#             showPlot, show the figure (it is always saved to resultsDir).
# Returns: nothing.
def plotSuccessHistogram(summaryDf, dirtyCount, showPlot=False):
	data = summaryDf[summaryDf["InitialDirty"] == dirtyCount]
	counts = sorted(data["numAgents"].unique())
	barWidth = 0.38
	positions = np.arange(len(counts))

	fig, ax = plt.subplots(figsize=(8, 5))
	for offset, mode in zip((-barWidth / 2, barWidth / 2), MODES):
		modeData = data[data["coordination"] == mode].set_index("numAgents")
		values = [modeData.loc[k, "SuccessProbability"] for k in counts]
		bars = ax.bar(
			positions + offset,
			values,
			barWidth,
			color=MODE_COLORS[mode],
			label=MODE_LABELS[mode],
		)
		ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

	ax.set_xticks(positions)
	ax.set_xticklabels([str(k) for k in counts])
	ax.set_ylim(0, 1.25)
	ax.set_xlabel("Number of Robots")
	ax.set_ylabel("Success probability")
	ax.set_title(
		f"Success Probability ({dirtyCount} dirty cells, {MAX_STEPS} steps, {N_SEEDS} simulations)"
	)
	ax.legend(loc="upper center", ncol=2)
	ax.grid(axis="y", alpha=0.3)
	finishFigure(fig, resultsDir, f"success_probability_{dirtyCount}.png", showPlot)


# Purpose: line plot of the mean steps until clean (mean ± std) versus the number of robots, one panel per amount of dirt.
# Parameters: summaryDf, results table, showPlot, show the figure (it is always saved to resultsDir).
# Returns: nothing.
def plotTimeVsAgents(summaryDf, showPlot=False):
	dirtyCounts = sorted(summaryDf["InitialDirty"].unique())
	fig, axes = plt.subplots(
		1, len(dirtyCounts), figsize=(6 * len(dirtyCounts), 4.5), sharey=True
	)
	axes = np.atleast_1d(axes)
	for ax, dirtyCount in zip(axes, dirtyCounts):
		for mode in MODES:
			data = summaryDf[
				(summaryDf["InitialDirty"] == dirtyCount)
				& (summaryDf["coordination"] == mode)
			]
			ax.errorbar(
				data["numAgents"],
				data["MeanSteps"],
				yerr=data["StdSteps"],
				fmt="o-",
				color=MODE_COLORS[mode],
				capsize=3,
				label=MODE_LABELS[mode],
			)
		ax.axhline(MAX_STEPS, linestyle="--", color="#8a8a85", linewidth=1)
		ax.set_xscale("log", base=2)
		ax.set_xticks(AGENT_COUNTS)
		ax.set_xticklabels([str(k) for k in AGENT_COUNTS])
		ax.minorticks_off()
		ax.set_xlabel("Number of Robots")
		ax.set_title(f"{dirtyCount} dirty cells")
		ax.grid(alpha=0.3)
	axes[0].set_ylabel(f"Steps until clean (limit {MAX_STEPS})")
	axes[0].legend()
	fig.suptitle("Cleaning Time (mean ± std over simulations)")
	fig.tight_layout()
	finishFigure(fig, resultsDir, "time_vs_agents.png", showPlot)


# Purpose: draw the room at the current step of a model: dirty cells, robots and coordination graph edges.
# Parameters: ax, matplotlib axes to draw on, model, the CleaningModel, title, text above the panel.
# Returns: nothing.
def drawRoom(ax, model, title):
	dirty = model.grid.dirty.data
	ax.imshow(
		dirty.T,
		origin="lower",
		cmap=ListedColormap(["#f5f5f0", "#b5651d"]),
		vmin=0,
		vmax=1,
	)
	robots = list(model.agents)
	for i, j in buildEdges(robots, model.d):
		(x1, y1), (x2, y2) = robots[i].cell.coordinate, robots[j].cell.coordinate
		ax.plot([x1, x2], [y1, y2], "-", color=COLORS["PURPLE"], linewidth=1.5)
	xs = [r.cell.coordinate[0] for r in robots]
	ys = [r.cell.coordinate[1] for r in robots]
	ax.scatter(xs, ys, s=70, color=COLORS["BLUE"], edgecolor="white", zorder=3)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(f"{title} ({model.numDirty()} dirty)")


# Purpose: save a snapshot of one Max-Plus simulation at several steps.
# Parameters: showPlot, show the figure (it is always saved to resultsDir).
# Returns: nothing.
def plotSnapshot(showPlot=False):
	model = CleaningModel(
		WIDTH,
		HEIGHT,
		SNAPSHOT_AGENTS,
		MAX_STEPS,
		SNAPSHOT_DIRTY / (WIDTH * HEIGHT),
		seed=0,
		coordination="maxplus",
	)
	fig, axes = plt.subplots(1, len(SNAPSHOT_STEPS), figsize=(4 * len(SNAPSHOT_STEPS), 6.5))
	for ax, step in zip(axes, SNAPSHOT_STEPS):
		while model.stepCount < step and model.running:
			model.step()
		drawRoom(ax, model, f"Step {model.stepCount}")
	fig.suptitle(
		"Max-Plus simulation: brown = dirty, blue = robot, purple = coordination edge"
	)
	fig.tight_layout()
	finishFigure(fig, resultsDir, "snapshot.png", showPlot)


# Purpose: run the whole Activity 2 experiment: sweep, results table, CSV files and figures.
# Parameters: none.
# Returns: nothing.
def main():
	df = runActivity2Sweep()
	summaryDf = buildActivity2Summary(df)
	saveActivity2Results(df, summaryDf)
	printActivity2Table(summaryDf)
	for dirtyCount in DIRTY_COUNTS:
		plotSuccessHistogram(summaryDf, dirtyCount)
	plotTimeVsAgents(summaryDf)
	plotSnapshot()


if __name__ == "__main__":
	main()
