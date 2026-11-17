# Description: reactive cleaning agent. It cleans its cell if it is dirty, otherwise moves to a random neighboring cell.
# Authors: 
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-17

from mesa.discrete_space import CellAgent


class CleaningAgent(CellAgent):
	# The 8 possible offsets of the Moore neighborhood (staying in place is excluded).
	DIRECTIONS = tuple(
		(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
	)

	# Purpose: create the agent and place it in the starting cell (0, 0).
	# Parameters: model, the MESA model the agent belongs to.
	# Returns: nothing.
	def __init__(self, model):
		super().__init__(model)
		self.cell = model.grid[(0, 0)]
		self.moves = 0
		self.wallBumps = 0
		self.blockedMoves = 0

	# Purpose: run one step; clean the current cell if it is dirty, otherwise move.
	# Parameters: none.
	# Returns: nothing.
	def step(self):
		dirty = self.model.grid.dirty.data
		if dirty[self.cell.coordinate]:
			dirty[self.cell.coordinate] = False
		else:
			self.moveRandomly()

	# Purpose: try to move to one of the 8 neighboring cells chosen at random. Counts successful moves, wall bumps and moves blocked by another agent.
	# Parameters: none.
	# Returns: nothing.
	def moveRandomly(self):
		dx, dy = self.random.choice(self.DIRECTIONS)
		x, y = self.cell.coordinate
		width, height = self.model.grid.dimensions
		nx, ny = x + dx, y + dy
		if not (0 <= nx < width and 0 <= ny < height):
			self.wallBumps += 1
			return  # wall: stay in place
		target = self.model.grid[(nx, ny)]
		if not target.is_empty:
			self.blockedMoves += 1
			return  # if occupied by another agent, stay in place
		self.cell = target
		self.moves += 1
