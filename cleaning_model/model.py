import mesa
import numpy as np
from agents import CleaningAgent


class CleaningModel(mesa.Model):
    def __init__(self, width, height, n_agents, max_steps, dirty_pct):
        super().__init__()

        self.grid = mesa.discrete_space.OrthogonalMooreGrid(
            (width, height), capacity=None, random=self.random
        )

        self.max_steps = max_steps
        self.running = True

        self.n_cells = width * height
        self.init_dirty = round(dirty_pct * self.n_cells)
        self.grid.create_property_layer("dirty", default_value=False, dtype=bool)
        dirty_idxs = np.random.Generator.choice(
            self.n_cells, size=self.initial_dirty, replace=False
        )
        self.grid.dirty.data.flat[dirty_idxs] = True

        CleaningAgent.create_agents(self, n_agents)

        self.step_count = 0

        self.full_cleaning_time = 0 if (self.init_dirty == 0) else None

    def n_dirty(self):
        return int(self.grid.dirty.data.sum())

    def step(self):
        self.agents.shuffle_do("step")
        self.step_count += 1

        if self.full_cleaning_time is None and self.n_dirty == 0:
            self.full_cleaning_time = self.step_count

        if self.step_count >= self.max_steps:
            self.running = False

    def summary(self):
        pass

    def run(self):

        while self.running:
            self.step()

        return self.summary
