# Description: figures and tables for the Activity 2 report: flowchart of one simulation step, speedup of Max-Plus, failed moves and total moves compared with the random (reactive) robots.
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
from matplotlib.patches import FancyBboxPatch, Polygon

ScriptLocation = Path(__file__).resolve().parent
sys.path.insert(0, str(ScriptLocation))
sys.path.insert(0, str(ScriptLocation.parent / "cleaning_model"))

from run_experiments import ci95, drawEfficiency, drawSpeedup, finishFigure, resultsDir
from activity2 import AGENT_COUNTS, MODE_COLORS, MODE_LABELS, MODES


# Purpose: draw a rounded box with centered text on the flowchart.
# Parameters: ax, matplotlib axes, x, y, center of the box, w, h, width and height, text, label of the box,
#             color, background color of the box.
# Returns: nothing.
def drawBox(ax, x, y, w, h, text, color="#cde2fb"):
	box = FancyBboxPatch(
		(x - w / 2, y - h / 2),
		w,
		h,
		boxstyle="round,pad=0.005,rounding_size=0.015",
		facecolor=color,
		edgecolor="#2a78d6",
		linewidth=1.5,
	)
	ax.add_patch(box)
	ax.text(x, y, text, ha="center", va="center", fontsize=9)


# Purpose: draw a diamond (decision) with centered text on the flowchart.
# Parameters: ax, matplotlib axes, x, y, center of the diamond, w, h, width and height, text, label of the decision.
# Returns: nothing.
def drawDiamond(ax, x, y, w, h, text):
	points = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
	ax.add_patch(
		Polygon(points, closed=True, facecolor="#fbe3a8", edgecolor="#eda100", linewidth=1.5)
	)
	ax.text(x, y, text, ha="center", va="center", fontsize=9)


# Purpose: draw a line through a list of points with an arrow head on the last segment.
# Parameters: ax, matplotlib axes, points, list of (x, y) tuples, label, optional text next to the first segment.
# Returns: nothing.
def drawArrow(ax, points, label=None):
	xs, ys = zip(*points)
	ax.plot(xs[:-1], ys[:-1], color="#444444", linewidth=1.3, zorder=1)
	ax.annotate(
		"",
		xy=points[-1],
		xytext=points[-2],
		arrowprops={"arrowstyle": "-|>", "color": "#444444", "linewidth": 1.3},
	)
	if label:
		ax.text(
			(points[0][0] + points[1][0]) / 2,
			(points[0][1] + points[1][1]) / 2 + 0.012,
			label,
			fontsize=9,
			ha="center",
			color="#444444",
		)


# Purpose: draw and save the flowchart of ONE simulation step of the Max-Plus coordinated robots.
# Parameters: showPlot, show the figure (it is always saved to resultsDir as flowchart.png).
# Returns: nothing.
def plotFlowchart(showPlot=False):
	fig, ax = plt.subplots(figsize=(8, 11))
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.axis("off")

	drawBox(ax, 0.5, 0.95, 0.5, 0.05, "Start of step t", "#d9f2e6")
	drawBox(ax, 0.5, 0.87, 0.7, 0.06, "Build the coordination graph:\nedge between robots at Manhattan distance <= d")
	drawBox(ax, 0.5, 0.79, 0.7, 0.06, "Local utility q_i(a_i) of every robot\n(a: move to dirty, b: stay, c: move to clean)")
	drawBox(ax, 0.5, 0.71, 0.7, 0.06, "Pairwise payoff f_ij(a_i, a_j) of every edge\n(u: different cells, v: same cell, w: otherwise)")
	drawBox(
		ax,
		0.5,
		0.61,
		0.76,
		0.09,
		"Max-Plus message passing (at most 30 iterations)\nmu_ij(a_j) = max_ai [q_i + f_ij + sum_k mu_ki], damped\nstop when messages converge; keep the best joint action",
	)
	drawBox(ax, 0.5, 0.51, 0.7, 0.06, "Robots act in random order\n(a robot waits if its target cell is occupied)")
	drawDiamond(ax, 0.5, 0.40, 0.36, 0.10, "Robot on a\ndirty cell?")
	drawBox(ax, 0.15, 0.30, 0.26, 0.06, "Clean the cell")
	drawBox(ax, 0.84, 0.30, 0.30, 0.07, "Execute planned move\n(stay if wall or occupied)")
	drawBox(ax, 0.5, 0.19, 0.5, 0.06, "Record metrics\n(clean %, moves, blocked moves)")
	drawDiamond(ax, 0.5, 0.08, 0.40, 0.10, "All clean or\nt >= 500?")
	drawBox(ax, 0.86, 0.08, 0.18, 0.05, "End", "#f7c9c9")

	for top, bottom in [(0.925, 0.90), (0.84, 0.82), (0.76, 0.74), (0.68, 0.655), (0.565, 0.54), (0.48, 0.45)]:
		drawArrow(ax, [(0.5, top), (0.5, bottom)])
	drawArrow(ax, [(0.32, 0.40), (0.15, 0.40), (0.15, 0.33)], "yes")
	drawArrow(ax, [(0.68, 0.40), (0.84, 0.40), (0.84, 0.335)], "no")
	drawArrow(ax, [(0.15, 0.27), (0.15, 0.19), (0.25, 0.19)])
	drawArrow(ax, [(0.84, 0.265), (0.84, 0.19), (0.75, 0.19)])
	drawArrow(ax, [(0.5, 0.16), (0.5, 0.13)])
	drawArrow(ax, [(0.70, 0.08), (0.77, 0.08)], "yes")
	drawArrow(ax, [(0.30, 0.08), (0.03, 0.08), (0.03, 0.95), (0.25, 0.95)], "no: next step")

	ax.set_title("Flowchart of one simulation step (Max-Plus coordination)")
	finishFigure(fig, resultsDir, "flowchart.png", showPlot)


# Purpose: speedup T(1)/T(k) and efficiency (speedup / k) of one coordination mode, computed per seed (every seed is the same room for every robot count).
# Parameters: df, DataFrame with one row per simulation (activity2_sweep.csv), mode, "reactive" or "maxplus",
#             dirtyCount, initial number of dirty cells.
# Returns: DataFrame with one row per robot count: MeanSteps, Speedup, CI95Speedup, Efficiency, CI95Efficiency.
def buildSpeedupTable(df, mode, dirtyCount):
	data = df[(df["coordination"] == mode) & (df["InitialDirty"] == dirtyCount)]
	times = data.pivot_table(index="seed", columns="numAgents", values="Step")
	perSeedSpeedup = times.rdiv(times[times.columns.min()], axis=0)
	table = pd.DataFrame(
		{
			"numAgents": times.columns,
			"MeanSteps": times.mean().to_numpy(),
			"Speedup": perSeedSpeedup.mean().to_numpy(),
			"CI95Speedup": perSeedSpeedup.apply(ci95).to_numpy(),
		}
	)
	table["Efficiency"] = table["Speedup"] / table["numAgents"]
	table["CI95Efficiency"] = table["CI95Speedup"] / table["numAgents"]
	return table


# Purpose: plot the measured speedup against the ideal one, and the efficiency, for one table from buildSpeedupTable.
# Parameters: speedupTable, output of buildSpeedupTable, showPlot, show the figure (it is always saved as speedup_efficiency.png).
# Returns: nothing.
def plotSpeedupEfficiency(speedupTable, showPlot=False):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
	drawSpeedup(ax1, speedupTable, AGENT_COUNTS)
	drawEfficiency(ax2, speedupTable, AGENT_COUNTS)
	fig.tight_layout()
	finishFigure(fig, resultsDir, "speedup_efficiency.png", showPlot)


# Purpose: plot the mean total movements of all robots (mean ± std) versus the number of robots, one panel per amount of dirt.
# Parameters: summaryDf, results table from buildActivity2Summary, showPlot, show the figure (it is always saved as moves_vs_agents.png).
# Returns: nothing.
def plotMovesVsAgents(summaryDf, showPlot=False):
	dirtyCounts = sorted(summaryDf["InitialDirty"].unique())
	fig, axes = plt.subplots(1, len(dirtyCounts), figsize=(6 * len(dirtyCounts), 4.5), sharey=True)
	axes = np.atleast_1d(axes)
	for ax, dirtyCount in zip(axes, dirtyCounts):
		for mode in MODES:
			data = summaryDf[
				(summaryDf["InitialDirty"] == dirtyCount) & (summaryDf["coordination"] == mode)
			]
			ax.errorbar(
				data["numAgents"],
				data["MeanMoves"],
				yerr=data["StdMoves"],
				fmt="o-",
				color=MODE_COLORS[mode],
				capsize=3,
				label=MODE_LABELS[mode],
			)
		ax.set_yscale("log")
		ax.set_yticks([200, 500, 1_000, 2_000, 5_000])
		ax.set_yticklabels(["200", "500", "1,000", "2,000", "5,000"])
		ax.set_xscale("log", base=2)
		ax.set_xticks(AGENT_COUNTS)
		ax.set_xticklabels([str(k) for k in AGENT_COUNTS])
		ax.minorticks_off()
		ax.set_xlabel("Number of Robots")
		ax.set_title(f"{dirtyCount} dirty cells")
		ax.grid(alpha=0.3)
	axes[0].set_ylabel("Total movements, all robots (log scale)")
	axes[0].legend()
	fig.suptitle("Total Movements (mean ± std over simulations)")
	fig.tight_layout()
	finishFigure(fig, resultsDir, "moves_vs_agents.png", showPlot)


# Purpose: plot the failed move attempts per run (moves blocked by another robot and wall bumps) for both coordination modes.
# Parameters: df, DataFrame with one row per simulation (activity2_sweep.csv), dirtyCount, initial number of dirty cells,
#             showPlot, show the figure (it is always saved as failed_moves_comparison.png).
# Returns: nothing.
def plotFailedMoves(df, dirtyCount, showPlot=False):
	data = df[df["InitialDirty"] == dirtyCount]
	fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
	for ax, column, title in zip(
		axes,
		("BlockedMoves", "WallBumps"),
		("Moves blocked by another robot", "Moves into a wall"),
	):
		for mode in MODES:
			grouped = data[data["coordination"] == mode].groupby("numAgents")[column]
			ax.errorbar(
				grouped.mean().index,
				grouped.mean(),
				yerr=grouped.apply(ci95),
				fmt="o-",
				color=MODE_COLORS[mode],
				capsize=3,
				label=MODE_LABELS[mode],
			)
		ax.set_xscale("log", base=2)
		ax.set_xticks(AGENT_COUNTS)
		ax.set_xticklabels([str(k) for k in AGENT_COUNTS])
		ax.minorticks_off()
		ax.set_xlabel("Number of Robots")
		ax.set_ylabel("Failed moves per run (mean, 95% CI)")
		ax.set_title(title)
		ax.grid(alpha=0.3)
	axes[0].legend()
	fig.suptitle(f"Failed Move Attempts ({dirtyCount} dirty cells)")
	fig.tight_layout()
	finishFigure(fig, resultsDir, "failed_moves_comparison.png", showPlot)
