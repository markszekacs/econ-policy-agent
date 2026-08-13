"""Aggregate metrics computed from logged runs."""

import json
import statistics
from pathlib import Path


def load_runs(log_dir: Path) -> list[dict]:
    """Load all JSON run logs from a directory."""
    runs = []
    for path in sorted(log_dir.glob("**/*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            continue
    return runs


def agent_confidence_variance(runs: list[dict]) -> dict:
    """
    Per-agent self_confidence variance across runs.
    Returns {agent_name: {"mean": float, "std": float, "values": list}}
    """
    agent_names = [
        "macroeconomist", "labor_economist",
        "trade_unionist", "institutional", "fiscal_expert"
    ]
    result = {}
    for agent in agent_names:
        values = []
        for run in runs:
            agent_log = run.get(agent, {})
            conf = agent_log.get("self_confidence")
            if conf is not None:
                values.append(conf)
        if values:
            result[agent] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return result


def evidence_strength_by_query(runs: list[dict]) -> dict:
    """
    Average evidence_strength per query across all agents.
    Returns {query_id: {"mean": float, "by_agent": dict}}
    """
    result = {}
    for run in runs:
        qid = run.get("query", "unknown")[:40]
        agent_names = [
            "macroeconomist", "labor_economist",
            "trade_unionist", "institutional", "fiscal_expert"
        ]
        by_agent = {}
        for agent in agent_names:
            es = run.get(agent, {}).get("evidence_strength")
            if es is not None:
                by_agent[agent] = es
        if by_agent:
            result[qid] = {
                "mean": statistics.mean(by_agent.values()),
                "by_agent": by_agent,
            }
    return result


def prior_knowledge_rates(runs: list[dict]) -> dict:
    """
    Per-agent average prior_knowledge_rate.
    Returns {agent_name: float}
    """
    agent_names = [
        "macroeconomist", "labor_economist",
        "trade_unionist", "institutional", "fiscal_expert"
    ]
    result = {}
    for agent in agent_names:
        values = [
            run.get(agent, {}).get("prior_knowledge_rate")
            for run in runs
            if run.get(agent, {}).get("prior_knowledge_rate") is not None
        ]
        if values:
            result[agent] = statistics.mean(values)
    return result


def critic_issue_taxonomy(runs: list[dict]) -> dict:
    """
    Count of each issue_type across all critic outputs.
    Returns {issue_type: count}
    """
    taxonomy = {}
    for run in runs:
        issues = run.get("critic", {}).get("issues", [])
        for issue in issues:
            issue_type = issue.get("issue_type", "unknown")
            taxonomy[issue_type] = taxonomy.get(issue_type, 0) + 1
    return taxonomy


def latency_stats(runs: list[dict]) -> dict:
    """
    Total latency statistics across runs.
    Returns {"mean_ms": float, "p50_ms": float, "p95_ms": float}
    """
    latencies = [
        run.get("total_latency_ms")
        for run in runs
        if run.get("total_latency_ms")
    ]
    if not latencies:
        return {}
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": sorted_lat[n // 2],
        "p95_ms": sorted_lat[int(n * 0.95)],
        "min_ms": sorted_lat[0],
        "max_ms": sorted_lat[-1],
    }
