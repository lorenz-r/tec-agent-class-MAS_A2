from pathlib import Path
from model import CleaningModel

#Testing for data collection
# Location of script
SCRIPT_LOCATION = Path(__file__).parent
ROOT = SCRIPT_LOCATION.parent
RESULTS_DIR = ROOT / "results"

# Constants
WIDTH = 10
HEIGHT = 10
N_AGENTS = 5
MAX_STEPS = 500
DIRTY_PCT = 0.3

if __name__ == "__main__":
    model = CleaningModel(
        width=WIDTH,
        height=HEIGHT,
        n_agents=N_AGENTS,
        max_steps=MAX_STEPS,
        dirty_pct=DIRTY_PCT,
    )
    results = model.run()
    print(results)
    model.datacollector.get_agent_vars_dataframe().to_csv(RESULTS_DIR / "agent_test_results.csv")
    model.datacollector.get_model_vars_dataframe().to_csv(RESULTS_DIR / "model_test_results.csv", index_label="Step"
                                                          