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

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "CleaningPercentage": lambda m: 100 * (1 - m.n_dirty() / m.n_cells),
                "DirtyCells": self.n_dirty,
                "TotalMoves": lambda m: m.agents.agg("moves", sum),
                "FullCleaningTime": "full_cleaning_time",
            },
            agent_reporters={
                "Moves": "moves",
            },
        )
        # Initial state
        self.datacollector.collect(self)

    def n_dirty(self):
        return int(self.grid.dirty.data.sum())

    def step(self):
        self.agents.shuffle_do("step")
        self.step_count += 1

        if self.full_cleaning_time is None and self.n_dirty == 0:
            self.full_cleaning_time = self.step_count

        if self.step_count >= self.max_steps:
            self.running = False

        self.datacollector.collect(self)

    def summary(self):
        summary_content = {
            "CleaningPercentage": 100 * (1 - self.n_dirty() / self.n_cells),
            "TotalMoves": self.agents.agg("moves", sum),
            "FullCleaningTime": self.full_cleaning_time,
        }

        return summary_content

    def run(self):

        while self.running:
            self.step()

        return self.summary
