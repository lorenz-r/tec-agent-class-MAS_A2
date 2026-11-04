Given:

- Room of MxN spaces.
- Number of agents.
- Percentage of initially dirty cells.
- Maximum execution time.


Run the following simulation:

- Initialize dirty cells (random locations).
- All agents start at cell [1,1].
- At each time step:
    -  If the cell is dirty, the agent cleans it.
    - If the cell is clear, the agent chooses a random direction to move (one of the 8 neighboring cells) and selects the movement action (if it cannot move there, it remains in the same cell).
- Execute until the maximum time is reached.


You will need to collect the following information during execution:

- Time required until all cells are clean (or the maximum time is reached).
- Percentage of clean cells after the simulation.
- Number of movements made by all agents.