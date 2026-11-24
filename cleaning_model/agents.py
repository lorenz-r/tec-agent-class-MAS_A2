# Description: cleaning agent. It cleans its cell if it is dirty. Otherwise it moves: either to a random
#              neighboring cell (reactive mode) or to the cell chosen jointly by Max-Plus over the
#              coordination graph (maxplus mode, see coordination.py and maxplus.py).
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
    # Action set used by Max-Plus: index 0 = stay, indices 1..8 = DIRECTIONS.
    STAY = 0
    ACTIONS = ((0, 0),) + DIRECTIONS

    # Purpose: create the agent and place it in its starting cell.
    # Parameters: model, the MESA model the agent belongs to,
    #             cell, starting cell (None = cell (0, 0)).
    # Returns: nothing.
    def __init__(self, model, cell=None):
        super().__init__(model)
        self.cell = cell if cell is not None else model.grid[(0, 0)]
        self.moves = 0
        self.wallBumps = 0
        self.blockedMoves = 0
        self.cellsCleaned = 0
        self.plannedAction = self.STAY

    # Purpose: run one step; clean the current cell if it is dirty, otherwise move.
    # Parameters: none.
    # Returns: nothing.
    def step(self):
        dirty = self.model.grid.dirty.data
        if dirty[self.cell.coordinate]:
            dirty[self.cell.coordinate] = False
            self.cellsCleaned += 1
        elif self.model.coordination == "maxplus":
            self.tryMove(*self.ACTIONS[self.plannedAction])
        else:
            self.moveRandomly()

    # Purpose: try to move to one of the 8 neighboring cells chosen at random.
    # Parameters: none.
    # Returns: nothing.
    def moveRandomly(self):
        dx, dy = self.random.choice(self.DIRECTIONS)
        self.tryMove(dx, dy)

    # Purpose: try to move by the offset (dx, dy). Counts successful moves, wall bumps and moves blocked by another agent.
    # Parameters: dx, dy, offset of the move ((0, 0) = stay).
    # Returns: nothing.
    def tryMove(self, dx, dy):
        if (dx, dy) == (0, 0):
            return
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

    # Purpose: coordinate the agent would end up in after taking an action.
    # Parameters: action, index into ACTIONS.
    # Returns: tuple (x, y).
    def targetOf(self, action):
        dx, dy = self.ACTIONS[action]
        x, y = self.cell.coordinate
        return (x + dx, y + dy)
