"""Save and retrieve runs from PostgreSQL/SQLite."""

from datetime import datetime
from sqlalchemy.orm import Session
from database.models import RunRecord, AgentResult, CriticResult

AGENT_NAMES = [
    "macroeconomist", "labor_economist",
    "trade_unionist", "institutional", "fiscal_expert"
]

AGENT_OUTPUT_KEYS = {
    "macroeconomist": "macroeconomist_output",
    "labor_economist": "labor_economist_output",
    "trade_unionist": "trade_unionist_output",
    "institutional": "institutional_output",
    "fiscal_expert": "fiscal_output",
}


def save_run_to_db(state: dict, db: Session) -> RunRecord:
    """
    Persists a completed pipeline run to the database.
    Called from logger_node alongside JSON logging.
    """
    meta = state["metadata"]
    config = meta["config"]

    run = RunRecord(
        run_id=meta["run_id"],
        query=state["query"],
        ablation_mode=state.get("ablation_mode", "full"),
        model_id=config.macroeconomist.model_id,
        temperature=config.macroeconomist.temperature,
        top_k=config.retrieval.top_k,
        chunk_size=config.retrieval.chunk_size,
        prompt_variant=meta.get("prompt_variant", "baseline"),
        run_type=meta.get("run_type", "production"),
        experiment_id=meta.get("experiment_id"),
        experiment_type=meta.get("experiment_type"),
        final_output=state.get("final_output"),
        total_latency_ms=meta.get("total_latency_ms"),
        successful_agents=meta.get("successful_agents", 0),
        failed_agents=meta.get("failed_agents", []),
        skipped_agents=meta.get("skipped_agents", []),
        timestamp=datetime.fromisoformat(meta["timestamp"]),
        completed_at=datetime.utcnow(),
    )

    db.add(run)

    for agent_name in AGENT_NAMES:
        output_key = AGENT_OUTPUT_KEYS[agent_name]
        output = state.get(output_key) or {}
        agent_result = AgentResult(
            run_id=meta["run_id"],
            agent_name=agent_name,
            response=output.get("response"),
            self_confidence=output.get("self_confidence"),
            confidence_reasoning=output.get("confidence_reasoning"),
            evidence_strength=output.get("evidence_strength"),
            prior_knowledge_rate=output.get("prior_knowledge_rate"),
            latency_ms=output.get("latency_ms"),
            status=output.get("status", "skipped"),
            error=output.get("error"),
            retry_count=output.get("retry_count", 0),
            supported_claims=output.get("supported_claims", []),
        )
        db.add(agent_result)

    critic_output = state.get("critic_output") or {}
    critic_result = CriticResult(
        run_id=meta["run_id"],
        issues=critic_output.get("issues", []),
        overall_quality=critic_output.get("overall_quality"),
        confidence_adjustments=critic_output.get(
            "confidence_adjustments", {}
        ),
        reasoning=critic_output.get("reasoning"),
        status=critic_output.get("status", "skipped"),
        error=critic_output.get("error"),
        latency_ms=critic_output.get("latency_ms"),
    )
    db.add(critic_result)
    db.commit()
    db.refresh(run)
    return run


def get_run_by_id(run_id: str, db: Session) -> RunRecord | None:
    return db.query(RunRecord).filter(
        RunRecord.run_id == run_id
    ).first()


def get_recent_runs(
    db: Session,
    limit: int = 20,
    run_type: str | None = None,
) -> list[RunRecord]:
    q = db.query(RunRecord)
    if run_type:
        q = q.filter(RunRecord.run_type == run_type)
    return q.order_by(RunRecord.timestamp.desc()).limit(limit).all()
