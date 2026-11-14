import mesa
from ..cleaning_model.model import CleaningModel
import pandas as pd
from pathlib import Path


def run_experiments():
    results = mesa.batch_run(
        CleaningModel,
        parameters={
            "width": 20,
            "height": 20,
            "dirty_pct": 0.5,
            "max_steps": 50_000,
            "n_agents": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
            "seed": range(30),
        },
        iterations=1,
        data_collection_period=-1,
        number_processes=None,
    )

    df = pd.DataFrame(results)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sweep.csv", index=False)


if __name__ == "__main__":
    run_experiments()
