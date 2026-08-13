"""
Calibration experiment.
Runs each benchmark query 5 times at fixed temperature.
Measures self_confidence and evidence_strength variance.
Computes reliability metrics.
"""

import uuid
import time
import json
import statistics
from pathlib import Path

from eval.benchmark import BENCHMARK_QUERIES
from eval.judges.llm_judge import score_agent_quality
from graph.runner import run_query
from config import DEFAULT_CONFIG

CALIBRATION_RUNS = 5
CALIBRATION_TEMPERATURE = 0.3

AGENT_NAMES = [
    "macroeconomist", "labor_economist",
    "trade_unionist", "institutional", "fiscal_expert"
]


def run_calibration_experiment(
    query_ids: list[str] | None = None,
    n_runs: int = CALIBRATION_RUNS,
) -> dict:
    """
    Run calibration experiment.
    For each query: runs n_runs times, measures variance
    in self_confidence and evidence_strength per agent.
    Also runs LLM judge on each response.

    Returns calibration results dict.
    """
    queries = BENCHMARK_QUERIES
    if query_ids:
        queries = [q for q in queries if q["id"] in query_ids]

    experiment_id = str(uuid.uuid4())[:8]
    total = len(queries) * n_runs
    completed = 0
    results = {}

    print(f"Calibration experiment — {experiment_id}")
    print(f"Queries: {len(queries)} | Runs each: {n_runs}")
    print(f"Total runs: {total}\n")

    for benchmark in queries:
        qid = benchmark["id"]
        results[qid] = {
            agent: {
                "self_confidence": [],
                "evidence_strength": [],
                "judge_quality": [],
            }
            for agent in AGENT_NAMES
        }

        for run_idx in range(n_runs):
            completed += 1
            print(
                f"[{completed}/{total}] "
                f"{qid} run {run_idx + 1}/{n_runs}"
            )

            try:
                state = run_query(
                    query=benchmark["query"],
                    ablation_mode="full",
                    config=DEFAULT_CONFIG,
                    experiment_id=experiment_id,
                    experiment_type="calibration",
                    run_type="experiment",
                )

                for agent in AGENT_NAMES:
                    output_key = (
                        f"{agent}_output"
                        if agent != "fiscal_expert"
                        else "fiscal_output"
                    )
                    output = state.get(output_key, {}) or {}

                    if output.get("status") not in (
                        "success", "retry_success"
                    ):
                        continue

                    conf = output.get("self_confidence")
                    es = output.get("evidence_strength")
                    response = output.get("response", "")

                    if conf is not None:
                        results[qid][agent]["self_confidence"].append(conf)
                    if es is not None:
                        results[qid][agent]["evidence_strength"].append(es)

                    # LLM judge scoring
                    judge_score = score_agent_quality(
                        query=benchmark["query"],
                        agent_name=agent,
                        agent_response=response,
                        evidence_strength=es,
                    )
                    if judge_score:
                        results[qid][agent]["judge_quality"].append(
                            judge_score.overall
                        )

            except Exception as e:
                print(f"  ERROR: {e}")

            time.sleep(1.5)

    # Compute summary statistics
    summary = _compute_calibration_summary(results)

    # Save results
    out_path = (
        Path("logs/experiments/calibration") /
        f"{experiment_id}_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCalibration complete. Summary: {out_path}")
    return summary


def _compute_calibration_summary(results: dict) -> dict:
    """Compute variance and calibration metrics from raw results."""
    summary = {}

    for qid, agent_data in results.items():
        summary[qid] = {}
        for agent, metrics in agent_data.items():
            conf_values = metrics["self_confidence"]
            es_values = metrics["evidence_strength"]
            judge_values = metrics["judge_quality"]

            summary[qid][agent] = {
                "self_confidence": {
                    "mean": (statistics.mean(conf_values)
                             if conf_values else None),
                    "std": (statistics.stdev(conf_values)
                            if len(conf_values) > 1 else 0.0),
                    "values": conf_values,
                },
                "evidence_strength": {
                    "mean": (statistics.mean(es_values)
                             if es_values else None),
                    "std": (statistics.stdev(es_values)
                            if len(es_values) > 1 else 0.0),
                    "values": es_values,
                },
                "judge_quality": {
                    "mean": (statistics.mean(judge_values)
                             if judge_values else None),
                    "std": (statistics.stdev(judge_values)
                            if len(judge_values) > 1 else 0.0),
                    "values": judge_values,
                },
                # Key calibration metric:
                # how far is self_confidence from judge quality?
                "confidence_bias": (
                    statistics.mean(conf_values) -
                    statistics.mean(judge_values)
                    if conf_values and judge_values else None
                ),
            }

    return summary


if __name__ == "__main__":
    run_calibration_experiment(
        query_ids=["us_ubi", "eu_carbon_tax"],
        n_runs=3,
    )
