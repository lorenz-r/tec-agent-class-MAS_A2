import mesa
import numpy as np
from agents import CleaningAgent


class CleaningModel(mesa.Model):
    def __init__(self, width, height, n_agents, max_steps, dirty_pct):
        super().__init__()

        self.grid = mesa.discrete_space.OrthogonalMooreGrid(
            (width, height), capacity=None, random=self.random
        )

        self.n_cells = width * height
        self.init_dirty = round(dirty_pct * self.n_cells)
        self.grid.create_property_layer("dirty", default_value=False, dtype=bool)
        dirty_idxs = np.random.Generator.choice(
            self.n_cells, size=self.initial_dirty, replace=False
        )
        self.grid.dirty.data.flat[dirty_idxs] = True

        CleaningAgent.create_agents(self, n_agents)

        self.step_count = 0

    def step(self):
        self.agents.shuffle_do("step")
        self.step_count += 1
