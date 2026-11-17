# Description: runs a single simulation and exports the model and agent metrics to CSV.
# Authors: 
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-17

from pathlib import Path
from model import CleaningModel

SCRIPT_LOCATION = Path(__file__).parent
ROOT = SCRIPT_LOCATION.parent
RESULTS_DIR = ROOT / "results"

WIDTH = 10
HEIGHT = 10
N_AGENTS = 5
MAX_STEPS = 500
DIRTY_PCT = 0.3

if __name__ == "__main__":
    model = CleaningModel(
        width=WIDTH,
        height=HEIGHT,
        numAgents=N_AGENTS,
        maxSteps=MAX_STEPS,
        dirtyPct=DIRTY_PCT,
    )
    results = model.run()
    print(results)
    model.datacollector.get_agent_vars_dataframe().to_csv(RESULTS_DIR / "agent_test_results.csv")
    model.datacollector.get_model_vars_dataframe().to_csv(RESULTS_DIR / "model_test_results.csv", index_label="Step")
