"""logger_node: last graph node — saves RunLog to disk, returns empty dict."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from graph.state import AgentState
from eval.models import RunLog, AgentLog, CriticLog, SupportedClaimLog
from database.connection import SessionLocal
from database.persistence import save_run_to_db

LOGS_DIR = Path(__file__).parent.parent / "logs"


def _agent_log_from_output(agent_name: str, output: dict | None) -> AgentLog:
    if output is None:
        return AgentLog(
            agent_name=agent_name,
            response=None,
            self_confidence=None,
            confidence_reasoning=None,
            supported_claims=[],
            prior_knowledge_rate=None,
            latency_ms=None,
            status="skipped",
            error=None,
            retry_count=0,
        )
    return AgentLog(
        agent_name=agent_name,
        response=output.get("response"),
        self_confidence=output.get("self_confidence"),
        confidence_reasoning=output.get("confidence_reasoning"),
        supported_claims=[
            SupportedClaimLog(**c) for c in output.get("supported_claims", [])
        ],
        prior_knowledge_rate=output.get("prior_knowledge_rate"),
        evidence_strength=output.get("evidence_strength"),
        latency_ms=output.get("latency_ms"),
        status=output.get("status", "skipped"),
        error=output.get("error"),
        retry_count=output.get("retry_count", 0),
    )


def _critic_log_from_output(output: dict | None) -> CriticLog:
    if output is None:
        return CriticLog(
            issues=[],
            overall_quality=None,
            confidence_adjustments={},
            reasoning=None,
            status="skipped",
            error=None,
            latency_ms=None,
        )
    return CriticLog(
        issues=output.get("issues", []),
        overall_quality=output.get("overall_quality"),
        confidence_adjustments=output.get("confidence_adjustments", {}),
        reasoning=output.get("reasoning"),
        status=output.get("status", "skipped"),
        error=output.get("error"),
        latency_ms=output.get("latency_ms"),
    )


def _compute_total_latency(state: AgentState) -> int:
    """max(parallel agent latencies) + retrieval + synthesizer."""
    agent_latencies = []
    for key in [
        "macroeconomist_output", "labor_economist_output",
        "trade_unionist_output", "institutional_output", "fiscal_output",
    ]:
        out = state.get(key)
        if out and out.get("latency_ms"):
            agent_latencies.append(out["latency_ms"])

    parallel_ms = max(agent_latencies) if agent_latencies else 0
    retrieval_ms = state["metadata"].get("retrieval_latency_ms", 0)
    synthesizer_ms = state["metadata"].get("synthesizer_latency_ms") or 0
    critic_ms = (state.get("critic_output") or {}).get("latency_ms") or 0

    return retrieval_ms + parallel_ms + critic_ms + synthesizer_ms


def _save_run_log(log: RunLog, state: AgentState) -> None:
    meta = state["metadata"]
    run_type = meta.get("run_type", "production")
    experiment_type = meta.get("experiment_type")
    run_id = log.run_id

    if run_type == "production":
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = LOGS_DIR / "production" / f"{date_str}_run_{run_id}.json"
    else:
        exp_dir = {
            "temp_sweep": "temp_sweep",
            "critic_ablation": "critic_ablation",
            "prompt_robustness": "prompt_robustness",
        }.get(experiment_type or "", "misc")
        exp_id = meta.get("experiment_id", "exp")
        path = LOGS_DIR / "experiments" / exp_dir / f"{exp_id}_run_{run_id}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log.model_dump(), f, indent=2, default=str)


def logger_node(state: AgentState) -> dict:
    meta = state["metadata"]
    config = meta["config"]
    completed_at = datetime.now(timezone.utc).isoformat()

    log = RunLog(
        run_id=meta["run_id"],
        experiment_id=meta.get("experiment_id"),
        run_type=meta.get("run_type", "production"),
        experiment_type=meta.get("experiment_type"),
        query=state["query"],
        ablation_mode=state.get("ablation_mode", "full"),
        # Note: model_id and temperature reflect macroeconomist
        # config. For per-agent model info see agent logs.
        model_id=config.macroeconomist.model_id,
        temperature=config.macroeconomist.temperature,
        top_k=config.retrieval.top_k,
        chunk_size=config.retrieval.chunk_size,
        prompt_variant=meta.get("prompt_variant", "baseline"),
        retrieved_docs=state.get("retrieved_docs", []),
        macroeconomist=_agent_log_from_output("macroeconomist", state.get("macroeconomist_output")),
        labor_economist=_agent_log_from_output("labor_economist", state.get("labor_economist_output")),
        trade_unionist=_agent_log_from_output("trade_unionist", state.get("trade_unionist_output")),
        institutional=_agent_log_from_output("institutional", state.get("institutional_output")),
        fiscal_expert=_agent_log_from_output("fiscal_expert", state.get("fiscal_output")),
        critic=_critic_log_from_output(state.get("critic_output")),
        final_output=state.get("final_output"),
        synthesizer_latency_ms=meta.get("synthesizer_latency_ms"),
        total_latency_ms=_compute_total_latency(state),
        timestamp=meta["timestamp"],
        completed_at=completed_at,
        successful_agents=meta.get("successful_agents", 0),
        failed_agents=meta.get("failed_agents", []),
        skipped_agents=meta.get("skipped_agents", []),
        judge_scores=None,
    )

    _save_run_log(log, state)

    # Also persist to database (non-blocking)
    try:
        db = SessionLocal()
        save_run_to_db(state, db)
        db.close()
    except Exception:
        pass  # DB failure never breaks the pipeline

    return {}
