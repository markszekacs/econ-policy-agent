"""
Prompt robustness experiment.
Tests whether rephrasing a question changes agent outputs.
"""

import uuid
import time

from graph.runner import run_query
from config import DEFAULT_CONFIG

PROMPT_VARIANTS = {
    "us_ubi": {
        "baseline": "Should the United States implement Universal Basic Income?",
        "variant_1": "Evaluate the economic case for Universal Basic Income in the US.",
        "variant_2": "What are the arguments for and against UBI in the United States?",
        "variant_3": "Is UBI a good policy for the American economy?",
    },
    "us_tariffs": {
        "baseline": "What are the economic effects of Trump's tariff policy on the US economy?",
        "variant_1": "Analyze the macroeconomic impact of US import tariffs.",
        "variant_2": "Do tariffs help or hurt the American economy?",
        "variant_3": "Evaluate the costs and benefits of protectionist trade policy in the US.",
    },
    "eu_carbon_tax": {
        "baseline": "Should the EU introduce a universal carbon tax?",
        "variant_1": "Evaluate the economic effects of a carbon tax in the European Union.",
        "variant_2": "Is a carbon tax the right climate policy for Europe?",
        "variant_3": "What are the economic arguments for and against EU carbon pricing?",
    },
}

RUNS_PER_VARIANT = 2


def run_prompt_robustness(
    query_ids: list[str] | None = None,
    runs_per_variant: int = RUNS_PER_VARIANT,
) -> None:
    """
    Run prompt robustness experiment.
    Logs all runs to logs/experiments/prompt_robustness/.
    """
    variants = PROMPT_VARIANTS
    if query_ids:
        variants = {k: v for k, v in variants.items()
                   if k in query_ids}

    experiment_id = str(uuid.uuid4())[:8]
    total = sum(
        len(v) * runs_per_variant
        for v in variants.values()
    )
    completed = 0

    print(f"Prompt robustness — experiment {experiment_id}")
    print(f"Query groups: {len(variants)} | "
          f"Total runs: {total}\n")

    for query_id, query_variants in variants.items():
        for variant_name, query_text in query_variants.items():
            for run_idx in range(runs_per_variant):
                completed += 1
                print(
                    f"[{completed}/{total}] "
                    f"{query_id} | "
                    f"{variant_name} | "
                    f"run {run_idx + 1}"
                )

                try:
                    run_query(
                        query=query_text,
                        ablation_mode="full",
                        config=DEFAULT_CONFIG,
                        experiment_id=experiment_id,
                        experiment_type="prompt_robustness",
                        prompt_variant=variant_name,
                        run_type="experiment",
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")

                time.sleep(1.0)

    print(f"\nRobustness complete. "
          f"Logs in logs/experiments/prompt_robustness/")


if __name__ == "__main__":
    run_prompt_robustness(
        query_ids=["us_ubi"],
        runs_per_variant=1,
    )
