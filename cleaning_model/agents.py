from mesa.discrete_space import CellAgent


class CleaningAgent(CellAgent):
    
    def __init__(self, model):
        super().__init__(model)
        self.cell = model.grid[(0, 0)]
        self.moves = 0

    def step(self):
        if self.cell.dirty:
            self.cell.dirty = False
        else:
            self.moveRandomly()         