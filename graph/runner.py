"""Entry point for running a query through the full agent pipeline."""

import uuid
from datetime import datetime, timezone
from typing import Generator

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, AgentConfig
from graph.state import AgentState
from graph.graph import get_graph, EMPTY_CRITIC

load_dotenv()


def _build_initial_state(
    query: str,
    ablation_mode: str = "full",
    config: AgentConfig = DEFAULT_CONFIG,
    experiment_id: str | None = None,
    experiment_type: str | None = None,
    prompt_variant: str = "baseline",
    run_type: str = "production",
) -> AgentState:
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "query": query,
        "retrieved_docs": [],
        "macroeconomist_output": None,
        "labor_economist_output": None,
        "trade_unionist_output": None,
        "institutional_output": None,
        "fiscal_output": None,
        "critic_output": {**EMPTY_CRITIC},
        "final_output": None,
        "ablation_mode": ablation_mode,
        "metadata": {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "run_type": run_type,
            "experiment_type": experiment_type,
            "prompt_variant": prompt_variant,
            "config": config,
            "ablation_mode": ablation_mode,
            "temperature": config.macroeconomist.temperature,
            "timestamp": timestamp,
            "retrieval_latency_ms": 0,
            "synthesizer_latency_ms": None,
            "total_tokens": 0,
            "successful_agents": 0,
            "failed_agents": [],
            "skipped_agents": [],
            "fan_in_timestamp": None,
        },
    }


def run_query(
    query: str,
    ablation_mode: str = "full",
    config: AgentConfig = DEFAULT_CONFIG,
    experiment_id: str | None = None,
    experiment_type: str | None = None,
    prompt_variant: str = "baseline",
    run_type: str = "production",
) -> AgentState:
    initial_state = _build_initial_state(
        query, ablation_mode, config, experiment_id, experiment_type, prompt_variant, run_type
    )
    graph = get_graph()
    return graph.invoke(initial_state)


def stream_query(
    query: str,
    ablation_mode: str = "full",
    config: AgentConfig = DEFAULT_CONFIG,
) -> Generator[dict, None, None]:
    """Yield streaming graph updates for progressive UI rendering.

    Each yielded value: {node_name: partial_state_dict}
    Agent nodes yield as they complete (parallel fan-out gives one update per agent).
    """
    initial_state = _build_initial_state(query, ablation_mode, config)
    graph = get_graph()
    yield from graph.stream(initial_state, stream_mode="updates")
