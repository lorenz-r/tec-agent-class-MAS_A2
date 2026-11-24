# Description: coordination graph for the cleaning robots: payoff values, graph edges, local utilities
#              q_i(a_i) and pairwise payoffs f_ij(a_i, a_j) used by Max-Plus.
# Authors:
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-21

import numpy as np

# Utility used for impossible actions (moving through a wall). Stands in for -infinity.
NEG = -1e9

# Default values for the payoffs of the coordination graph.
DEFAULT_PAYOFFS = {
	"a": 10.0,  # move to a dirty cell
	"b": -1.0,  # stay
	"c": 0.0,  # move to a clean cell
	"u": 0.5,  # neighbors go to different cells (coverage)
	"v": -20.0,  # neighbors go to the same cell, both moving (congestion)
	"w": -20.0,  # otherwise (one moves into the cell the other stays in)
	"shaping": 2.0,  # bonus per step closer to the nearest dirty cell (added to c)
	"spread": 0.5,  # bonus per unit two neighbors move apart (added to u)
}


# Purpose: Manhattan distance between two coordinates.
# Parameters: p, r, tuples (x, y).
# Returns: integer distance.
def manhattan(p, r):
	return abs(p[0] - r[0]) + abs(p[1] - r[1])


# Purpose: edges of the coordination graph: robots i and j are connected if their Manhattan distance is <= d.
# Parameters: robots, list of CleaningAgent, d, maximum distance.
# Returns: list of tuples (i, j) with i < j.
def buildEdges(robots, d):
	return [
		(i, j)
		for i in range(len(robots))
		for j in range(i + 1, len(robots))
		if manhattan(robots[i].cell.coordinate, robots[j].cell.coordinate) <= d
	]


# Purpose: distance of every cell to the nearest dirty cell. Chebyshev distance, because agents move in the Moore neighborhood.
# Parameters: dirty, boolean array (width, height) with the dirty cells.
# Returns: numpy array of shape (width, height).
def distanceToDirty(dirty):
	width, height = dirty.shape
	dirtyCoords = np.argwhere(dirty)
	if len(dirtyCoords) == 0:
		return np.zeros((width, height))
	xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
	dist = np.maximum(
		np.abs(xs[..., None] - dirtyCoords[:, 0]),
		np.abs(ys[..., None] - dirtyCoords[:, 1]),
	)
	return dist.min(axis=-1).astype(float)


# Purpose: local utility q_i(a_i) of one robot for every action.
#          a -> move to a dirty cell, b -> stay, c -> move to a clean cell (+ shaping toward the
#          nearest dirty cell), NEG -> move through a wall. A robot standing on a dirty cell will
#          clean it this step, so for it only "stay" is allowed.
# Parameters: robot, the CleaningAgent, dirty, boolean array of dirty cells,
#             distToDirty, output of distanceToDirty, payoffs, dict with the payoff values.
# Returns: numpy array with one utility per action.
def localUtilities(robot, dirty, distToDirty, payoffs):
	width, height = dirty.shape
	x, y = robot.cell.coordinate
	q = np.full(len(robot.ACTIONS), NEG)
	q[robot.STAY] = payoffs["b"]
	if dirty[x, y]:
		return q  # cleaning this step, cannot move
	for action in range(1, len(robot.ACTIONS)):
		nx, ny = robot.targetOf(action)
		if not (0 <= nx < width and 0 <= ny < height):
			continue
		if dirty[nx, ny]:
			q[action] = payoffs["a"]
		else:
			q[action] = payoffs["c"] + payoffs["shaping"] * (
				distToDirty[x, y] - distToDirty[nx, ny]
			)
	# Tiny random term to break ties between equally good moves.
	q += 1e-3 * np.array([robot.random.random() for _ in robot.ACTIONS])
	return q


# Purpose: pairwise payoff f_ij(a_i, a_j) between two neighboring robots.
#          u -> different cells (+ spread bonus if they move apart), v -> same cell and both
#          moving (congestion), w -> otherwise (moving into the cell the other one stays in).
# Parameters: ri, rj, the two CleaningAgents, payoffs, dict with the payoff values.
# Returns: numpy array of shape (len(ACTIONS), len(ACTIONS)), rows = actions of ri.
def pairwisePayoff(ri, rj, payoffs):
	n = len(ri.ACTIONS)
	f = np.empty((n, n))
	d0 = manhattan(ri.cell.coordinate, rj.cell.coordinate)
	for ai in range(n):
		ti = ri.targetOf(ai)
		for aj in range(n):
			tj = rj.targetOf(aj)
			if ti != tj:
				f[ai, aj] = payoffs["u"] + payoffs["spread"] * (manhattan(ti, tj) - d0)
			elif ai != ri.STAY and aj != rj.STAY:
				f[ai, aj] = payoffs["v"]
			else:
				f[ai, aj] = payoffs["w"]
	return f
