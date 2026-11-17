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
    # Parameters: width, room width, height, room height, n_agents, number of robots,
    #             max_steps, maximum run time, dirty_pct, fraction of initially dirty cells (0 to 1),
    #             seed, random generator seed (None = random).
    # Returns: nothing.
    def __init__(self, width, height, n_agents, max_steps, dirty_pct, seed=None):
        super().__init__(rng=np.random.default_rng(seed=seed))

        self.grid = mesa.discrete_space.OrthogonalMooreGrid(
            (width, height), capacity=None, random=self.random
        )

        self.max_steps = max_steps
        self.running = True

        self.n_cells = width * height
        self.init_dirty = round(dirty_pct * self.n_cells)
        self.grid.create_property_layer("dirty", default_value=False, dtype=bool)
        dirty_idxs = self.rng.choice(self.n_cells, size=self.init_dirty, replace=False)
        self.grid.dirty.data.flat[dirty_idxs] = True

        CleaningAgent.create_agents(self, n_agents)

        self.step_count = 0

        self.full_cleaning_time = 0 if self.init_dirty == 0 else None

        if self.full_cleaning_time is not None:
            self.running = False

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "CleaningPercentage": lambda m: 100 * (1 - m.n_dirty() / m.n_cells),
                "DirtyCells": self.n_dirty,
                "TotalMoves": lambda m: m.agents.agg("moves", sum),
                "FullCleaningTime": "full_cleaning_time",
                "WallBumps": lambda m: m.agents.agg("wall_bumps", sum),
                "BlockedMoves": lambda m: m.agents.agg("blocked_moves", sum),
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
    def n_dirty(self):
        return int(self.grid.dirty.data.sum())

    # Purpose: advance the simulation one step: every agent acts, the stop state is updated and data is collected.
    # Parameters: none.
    # Returns: nothing.
    def step(self):
        self.agents.shuffle_do("step")
        self.step_count += 1

        self.updateFullCleaningTime()
        self.updateRunningState()

        self.datacollector.collect(self)

    # Purpose: record the step at which the room became completely clean (only the first time).
    # Parameters: none.
    # Returns: nothing.
    def updateFullCleaningTime(self):
        if self.full_cleaning_time is None and self.n_dirty() == 0:
            self.full_cleaning_time = self.step_count

    # Purpose: stop the simulation once everything is clean or max_steps is reached.
    # Parameters: none.
    # Returns: nothing.
    def updateRunningState(self):
        if self.full_cleaning_time is not None or self.step_count >= self.max_steps:
            self.running = False

    # Purpose: summarize the result of the run.
    # Parameters: none.
    # Returns: dictionary with CleaningPercentage (% of clean cells), TotalMoves (moves of all agents) and FullCleaningTime (step at which everything was clean, None if never).
    def summary(self):
        summary_content = {
            "CleaningPercentage": 100 * (1 - self.n_dirty() / self.n_cells),
            "TotalMoves": self.agents.agg("moves", sum),
            "FullCleaningTime": self.full_cleaning_time,
        }

        return summary_content

    # Purpose: run steps until everything is clean or max_steps is reached.
    # Parameters: none.
    # Returns: the dictionary from summary().
    def run(self):

        while self.running:
            self.step()

        return self.summary()
