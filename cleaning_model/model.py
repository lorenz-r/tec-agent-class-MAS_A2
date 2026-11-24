# Description: MESA model of a room with dirty cells and cleaning robots coordinated with a coordination
#              graph + Max-Plus message passing (Guestrin, 2002). Collects metrics with DataCollector.
# Authors:
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-17

import mesa
import numpy as np
from agents import CleaningAgent
from coordination import (
	DEFAULT_PAYOFFS,
	buildEdges,
	distanceToDirty,
	localUtilities,
	pairwisePayoff,
)
from maxplus import maxPlus


class CleaningModel(mesa.Model):
	# Purpose: build the room, dirty random cells, place the agents on random clean cells and set up the DataCollector.
	# Parameters: width, room width, height, room height, numAgents, number of robots,
	#             maxSteps, maximum run time, dirtyPct, fraction of initially dirty cells (0 to 1),
	#             seed, random generator seed (None = random),
	#             coordination, "maxplus" (coordination graph) or "reactive" (random moves),
	#             d, Manhattan distance for an edge in the coordination graph,
	#             maxPlusIters, maximum Max-Plus iterations per step, damping, message damping (0 to 1),
	#             payoffs, dict overriding entries of DEFAULT_PAYOFFS.
	# Returns: nothing.
	def __init__(
		self,
		width,
		height,
		numAgents,
		maxSteps,
		dirtyPct,
		seed=None,
		coordination="maxplus",
		d=4,
		maxPlusIters=30,
		damping=0.5,
		payoffs=None,
	):
		super().__init__(rng=np.random.default_rng(seed=seed))

		self.grid = mesa.discrete_space.OrthogonalMooreGrid(
			(width, height), capacity=None, random=self.random
		)

		self.maxSteps = maxSteps
		self.running = True

		self.coordination = coordination
		self.d = d
		self.maxPlusIters = maxPlusIters
		self.damping = damping
		self.payoffs = {**DEFAULT_PAYOFFS, **(payoffs or {})}
		self.edges = []
		self.maxPlusIterations = 0

		self.numCells = width * height
		self.initDirty = round(dirtyPct * self.numCells)
		self.grid.create_property_layer("dirty", default_value=False, dtype=bool)
		dirtyIdxs = self.rng.choice(self.numCells, size=self.initDirty, replace=False)
		self.grid.dirty.data.flat[dirtyIdxs] = True

		# Robots start at random clean cells, no two on the same cell.
		cleanIdxs = np.flatnonzero(~self.grid.dirty.data)
		startIdxs = self.rng.choice(cleanIdxs, size=numAgents, replace=False)
		startCells = [
			self.grid[np.unravel_index(i, (width, height))] for i in startIdxs
		]
		CleaningAgent.create_agents(self, numAgents, cell=startCells)

		self.stepCount = 0

		self.fullCleaningTime = 0 if self.initDirty == 0 else None

		if self.fullCleaningTime is not None:
			self.running = False

		self.datacollector = mesa.DataCollector(
			model_reporters={
				"CleaningPercentage": lambda m: 100 * (1 - m.numDirty() / m.numCells),
				"DirtyCells": self.numDirty,
				"TotalMoves": lambda m: m.agents.agg("moves", sum),
				"FullCleaningTime": "fullCleaningTime",
				"WallBumps": lambda m: m.agents.agg("wallBumps", sum),
				"BlockedMoves": lambda m: m.agents.agg("blockedMoves", sum),
				"GraphEdges": lambda m: len(m.edges),
				"MaxPlusIterations": "maxPlusIterations",
			},
			agent_reporters={
				"Moves": "moves",
				"CellsCleaned": "cellsCleaned",
			},
		)
		# Records step 0 (initial state).
		self.datacollector.collect(self)

	# Purpose: count the cells that are still dirty.
	# Parameters: none.
	# Returns: integer with the number of dirty cells.
	def numDirty(self):
		return int(self.grid.dirty.data.sum())

	# Purpose: advance the simulation one step: plan the joint action (Max-Plus), every agent acts,
	#          the stop state is updated and data is collected.
	# Parameters: none.
	# Returns: nothing.
	def step(self):
		if self.coordination == "maxplus":
			self.planJointAction()
			self.executeJointAction()
		else:
			self.agents.shuffle_do("step")
		self.stepCount += 1

		self.updateFullCleaningTime()
		self.updateRunningState()

		self.datacollector.collect(self)

	# Purpose: build the coordination graph (edge if Manhattan distance <= d), the q_i and f_ij tables,
	#          run Max-Plus and store the chosen action in each agent's plannedAction.
	# Parameters: none.
	# Returns: nothing.
	def planJointAction(self):
		robots = list(self.agents)
		dirty = self.grid.dirty.data
		distToDirty = distanceToDirty(dirty)
		q = [localUtilities(r, dirty, distToDirty, self.payoffs) for r in robots]
		self.edges = buildEdges(robots, self.d)
		f = [pairwisePayoff(robots[i], robots[j], self.payoffs) for i, j in self.edges]
		actions, self.maxPlusIterations = maxPlus(
			q, self.edges, f, self.maxPlusIters, self.damping
		)
		for robot, action in zip(robots, actions):
			robot.plannedAction = int(action)

	# Purpose: let every agent execute its planned action in random order, but an agent whose target
	#          cell is still occupied waits until the occupant has moved away. Without this, an agent
	#          following a teammate gets blocked just because it happened to act first. Agents that
	#          still cannot move (e.g. two agents swapping cells) act at the end and get blocked.
	# Parameters: none.
	# Returns: nothing.
	def executeJointAction(self):
		pending = list(self.agents.shuffle())
		progress = True
		while pending and progress:
			progress = False
			for agent in list(pending):
				target = agent.targetOf(agent.plannedAction)
				if agent.plannedAction == agent.STAY or self.grid[target].is_empty:
					agent.step()
					pending.remove(agent)
					progress = True
		for agent in pending:
			agent.step()

	# Purpose: record the step at which the room became completely clean (only the first time).
	# Parameters: none.
	# Returns: nothing.
	def updateFullCleaningTime(self):
		if self.fullCleaningTime is None and self.numDirty() == 0:
			self.fullCleaningTime = self.stepCount

	# Purpose: stop the simulation once everything is clean or maxSteps is reached.
	# Parameters: none.
	# Returns: nothing.
	def updateRunningState(self):
		if self.fullCleaningTime is not None or self.stepCount >= self.maxSteps:
			self.running = False

	# Purpose: summarize the result of the run.
	# Parameters: none.
	# Returns: dictionary with CleaningPercentage (% of clean cells), TotalMoves (moves of all agents) and FullCleaningTime (step at which everything was clean, None if never).
	def summary(self):
		summaryContent = {
			"CleaningPercentage": 100 * (1 - self.numDirty() / self.numCells),
			"TotalMoves": self.agents.agg("moves", sum),
			"FullCleaningTime": self.fullCleaningTime,
		}

		return summaryContent

	# Purpose: run steps until everything is clean or maxSteps is reached.
	# Parameters: none.
	# Returns: the dictionary from summary().
	def run(self):

		while self.running:
			self.step()

		return self.summary()
