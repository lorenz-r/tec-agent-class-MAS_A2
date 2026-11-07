from mesa import Agent

class CleaningAgent(Agent):
    def step(self):
        if self.pos in self.model.dirtyCells:
            self.model.dirtyCells.remove(self.pos)
        else:
            self.moveRandomly()