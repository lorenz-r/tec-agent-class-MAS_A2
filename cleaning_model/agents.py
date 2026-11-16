from mesa.discrete_space import CellAgent
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CleaningAgent(CellAgent):
    DIRECTIONS = tuple(
        (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
    )

    def __init__(self, model):
        super().__init__(model)
        self.cell = model.grid[(0, 0)]
        self.moves = 0

    def step(self):
        dirty = self.model.grid.dirty.data
        if dirty[self.cell.coordinate]:
            dirty[self.cell.coordinate] = False
        else:
            self.moveRandomly()

    def moveRandomly(self):
        dx, dy = self.random.choice(self.DIRECTIONS)
        x, y = self.cell.coordinate
        width, height = self.model.grid.dimensions
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            return  # wall: stay in place
        target = self.model.grid[(nx, ny)]
        if not target.is_empty:
            return  # occupied by another agent: stay in place
        self.cell = target
        self.moves += 1
