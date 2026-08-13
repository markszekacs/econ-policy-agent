"""
Temperature stability experiment.
Runs each benchmark query at 4 temperatures × 3 runs.
Total: 10 queries × 4 temps × 3 runs = 120 API calls.
"""

import uuid
import time
from dataclasses import replace

from eval.benchmark import BENCHMARK_QUERIES
from graph.runner import run_query
from config import DEFAULT_CONFIG

TEMPERATURES = [0.0, 0.2, 0.5, 0.7]
RUNS_PER_CONFIG = 3


def run_temperature_sweep(
    query_ids: list[str] | None = None,
    temperatures: list[float] = TEMPERATURES,
    runs_per_config: int = RUNS_PER_CONFIG,
) -> None:
    """
    Run temperature sweep experiment.
    Logs all runs to logs/experiments/temp_sweep/.

    Args:
        query_ids: subset of benchmark IDs to run.
                   None = all 10.
        temperatures: list of temperature values to sweep.
        runs_per_config: number of runs per temperature.
    """
    queries = BENCHMARK_QUERIES
    if query_ids:
        queries = [q for q in queries if q["id"] in query_ids]

    experiment_id = str(uuid.uuid4())[:8]
    total = len(queries) * len(temperatures) * runs_per_config
    completed = 0

    print(f"Temperature sweep — experiment {experiment_id}")
    print(f"Queries: {len(queries)} | Temps: {temperatures} "
          f"| Runs each: {runs_per_config}")
    print(f"Total API calls: {total}\n")

    for benchmark in queries:
        for temp in temperatures:
            config = replace(
                DEFAULT_CONFIG,
                macroeconomist=replace(
                    DEFAULT_CONFIG.macroeconomist,
                    temperature=temp
                ),
                labor_economist=replace(
                    DEFAULT_CONFIG.labor_economist,
                    temperature=temp
                ),
                trade_unionist=replace(
                    DEFAULT_CONFIG.trade_unionist,
                    temperature=temp
                ),
                institutional=replace(
                    DEFAULT_CONFIG.institutional,
                    temperature=temp
                ),
                fiscal_expert=replace(
                    DEFAULT_CONFIG.fiscal_expert,
                    temperature=temp
                ),
            )

            for run_idx in range(runs_per_config):
                completed += 1
                print(
                    f"[{completed}/{total}] "
                    f"{benchmark['id']} | "
                    f"temp={temp} | "
                    f"run {run_idx + 1}/{runs_per_config}"
                )

                try:
                    run_query(
                        query=benchmark["query"],
                        ablation_mode="full",
                        config=config,
                        experiment_id=experiment_id,
                        experiment_type="temp_sweep",
                        prompt_variant="baseline",
                        run_type="experiment",
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")

                time.sleep(1.0)

    print(f"\nSweep complete. Logs in logs/experiments/temp_sweep/")


if __name__ == "__main__":
    # Quick test: run 2 queries, 2 temps, 2 runs
    run_temperature_sweep(
        query_ids=["us_ubi", "us_tariffs"],
        temperatures=[0.2, 0.7],
        runs_per_config=2,
    )
