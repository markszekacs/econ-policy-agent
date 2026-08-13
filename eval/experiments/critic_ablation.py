"""
Critic ablation experiment.
Runs each benchmark query in 3 modes:
  single  = macroeconomist only
  multi   = all 5 agents, no critic
  full    = all 5 agents + critic
Total: 10 queries × 3 modes × 3 runs = 90 API calls.
"""

import uuid
import time

from eval.benchmark import BENCHMARK_QUERIES
from graph.runner import run_query
from config import DEFAULT_CONFIG

ABLATION_MODES = ["single", "multi", "full"]
RUNS_PER_MODE = 3


def run_critic_ablation(
    query_ids: list[str] | None = None,
    modes: list[str] = ABLATION_MODES,
    runs_per_mode: int = RUNS_PER_MODE,
) -> None:
    """
    Run critic ablation experiment.
    Logs all runs to logs/experiments/critic_ablation/.
    """
    queries = BENCHMARK_QUERIES
    if query_ids:
        queries = [q for q in queries if q["id"] in query_ids]

    experiment_id = str(uuid.uuid4())[:8]
    total = len(queries) * len(modes) * runs_per_mode
    completed = 0

    print(f"Critic ablation — experiment {experiment_id}")
    print(f"Queries: {len(queries)} | Modes: {modes} "
          f"| Runs each: {runs_per_mode}")
    print(f"Total API calls (approx): {total}\n")

    for benchmark in queries:
        for mode in modes:
            for run_idx in range(runs_per_mode):
                completed += 1
                print(
                    f"[{completed}/{total}] "
                    f"{benchmark['id']} | "
                    f"mode={mode} | "
                    f"run {run_idx + 1}/{runs_per_mode}"
                )

                try:
                    run_query(
                        query=benchmark["query"],
                        ablation_mode=mode,
                        config=DEFAULT_CONFIG,
                        experiment_id=experiment_id,
                        experiment_type="critic_ablation",
                        prompt_variant="baseline",
                        run_type="experiment",
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")

                time.sleep(1.0)

    print(f"\nAblation complete. Logs in logs/experiments/critic_ablation/")


if __name__ == "__main__":
    run_critic_ablation(
        query_ids=["us_ubi", "eu_carbon_tax"],
        modes=["multi", "full"],
        runs_per_mode=2,
    )
