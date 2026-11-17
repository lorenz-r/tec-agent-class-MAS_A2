# Description: MESA model of a room with dirty cells and reactive cleaning robots. Collects metrics with DataCollector.
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


class CleaningModel(mesa.Model):
    # Purpose: build the room, dirty random cells, place the agents and set up the DataCollector.
    # Parameters: width, room width, height, room height, numAgents, number of robots,
    #             maxSteps, maximum run time, dirtyPct, fraction of initially dirty cells (0 to 1),
    #             seed, random generator seed (None = random).
    # Returns: nothing.
    def __init__(self, width, height, numAgents, maxSteps, dirtyPct, seed=None):
        super().__init__(rng=np.random.default_rng(seed=seed))

        self.grid = mesa.discrete_space.OrthogonalMooreGrid(
            (width, height), capacity=None, random=self.random
        )

        self.maxSteps = maxSteps
        self.running = True

        self.numCells = width * height
        self.initDirty = round(dirtyPct * self.numCells)
        self.grid.create_property_layer("dirty", default_value=False, dtype=bool)
        dirtyIdxs = self.rng.choice(self.numCells, size=self.initDirty, replace=False)
        self.grid.dirty.data.flat[dirtyIdxs] = True

        CleaningAgent.create_agents(self, numAgents)

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
            },
            agent_reporters={
                "Moves": "moves",
            },
        )
        # Records step 0 (initial state).
        self.datacollector.collect(self)

    # Purpose: count the cells that are still dirty.
    # Parameters: none.
    # Returns: integer with the number of dirty cells.
    def numDirty(self):
        return int(self.grid.dirty.data.sum())

    # Purpose: advance the simulation one step: every agent acts, the stop state is updated and data is collected.
    # Parameters: none.
    # Returns: nothing.
    def step(self):
        self.agents.shuffle_do("step")
        self.stepCount += 1

        self.updateFullCleaningTime()
        self.updateRunningState()

        self.datacollector.collect(self)

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
